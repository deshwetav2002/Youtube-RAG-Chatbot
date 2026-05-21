import re
from langchain_community.document_loaders import YoutubeLoader
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()


# ── Helper: extract video ID from any YouTube URL format ─────────────────────
def extract_video_id(youtube_url: str) -> str:
    """
    Handles all common YouTube URL formats:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://youtu.be/VIDEO_ID
      https://www.youtube.com/embed/VIDEO_ID
    """
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",   # ?v=
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",  # youtu.be/
        r"(?:embed/)([a-zA-Z0-9_-]{11})",       # embed/
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {youtube_url}")


# ── 1. Load transcript using YouTube Data API v3 ─────────────────────────────
def load_transcript(youtube_url: str, youtube_api_key: str) -> list:
    """
    Uses YouTube Data API v3 to fetch video metadata (title, author),
    then youtube_transcript_api to fetch the actual transcript text.
    Returns a list of LangChain Document objects.
    """
    video_id = extract_video_id(youtube_url)

    # ── Fetch metadata via official Google API (not blocked by YouTube) ──────
    yt_service = build("youtube", "v3", developerKey=youtube_api_key)
    response = yt_service.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    if not response.get("items"):
        raise ValueError(f"No video found for ID: {video_id}")

    snippet = response["items"][0]["snippet"]
    title   = snippet.get("title", "Unknown Title")
    author  = snippet.get("channelTitle", "Unknown Channel")

    # ── Fetch transcript text ─────────────────────────────────────────────────
    transcript_list = YouTubeTranscriptApi.get_transcript(
        video_id,
        languages=["en", "en-US"]
    )

    # Merge all transcript segments into one text block
    full_text = " ".join(entry["text"] for entry in transcript_list)

    # Wrap in a LangChain Document with metadata
    doc = Document(
        page_content=full_text,
        metadata={
            "title":    title,
            "author":   author,
            "video_id": video_id,
            "source":   youtube_url,
        }
    )
    return [doc]


# ── 2. Chunk the transcript ───────────────────────────────────────────────────
def chunk_documents(docs: list, chunk_size: int = 1000, chunk_overlap: int = 150) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# ── 3. Build FAISS vector store ───────────────────────────────────────────────
def build_vectorstore(chunks: list) -> FAISS:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.from_documents(chunks, embedding)


# ── 4. Format retrieved docs into a single string ────────────────────────────
def format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ── 5. Format chat history as plain text ─────────────────────────────────────
def format_history(chat_history: list) -> str:
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


# ── 6. Build LCEL RAG chain ───────────────────────────────────────────────────
def build_rag_chain(vectorstore: FAISS, hf_api_token: str):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    endpoint = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        huggingfacehub_api_token=hf_api_token,
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and a follow-up question, rephrase the "
         "follow-up into a standalone question. Return ONLY the rephrased question."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert assistant that answers questions strictly based on "
         "the YouTube video transcript provided below.\n\n"
         "Rules:\n"
         "- Only use the transcript context to answer.\n"
         "- If the answer is not in the transcript, say: "
         "'I couldn't find that information in this video.'\n"
         "- Be concise and accurate.\n\n"
         "Transcript context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    parser = StrOutputParser()
    condense_chain = condense_prompt | llm | parser
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        )
        | answer_prompt | llm | parser
    )

    return {
        "condense_chain": condense_chain,
        "rag_chain": rag_chain,
        "retriever": retriever
    }


# ── 7. Ask a question ─────────────────────────────────────────────────────────
def ask_question(chain_bundle: dict, question: str, chat_history: list) -> dict:
    condense_chain = chain_bundle["condense_chain"]
    rag_chain      = chain_bundle["rag_chain"]
    retriever      = chain_bundle["retriever"]

    if chat_history:
        standalone_question = condense_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        standalone_question = standalone_question.strip().splitlines()[0]
    else:
        standalone_question = question

    source_docs = retriever.invoke(standalone_question)

    answer = rag_chain.invoke({
        "standalone_question": standalone_question,
        "question": question,
        "chat_history": chat_history
    })

    return {"answer": answer.strip(), "source_documents": source_docs}


# ── 8. One-shot convenience function ─────────────────────────────────────────
def process_youtube_url(youtube_url: str, hf_api_token: str, youtube_api_key: str) -> tuple:
    docs         = load_transcript(youtube_url, youtube_api_key)
    chunks       = chunk_documents(docs)
    vectorstore  = build_vectorstore(chunks)
    chain_bundle = build_rag_chain(vectorstore, hf_api_token)
    metadata     = docs[0].metadata if docs else {}
    return chain_bundle, metadata
