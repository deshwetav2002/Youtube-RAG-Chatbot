from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

#--1. Load transcript from Youtube URL--

def load_transcript(youtube_url: str) -> list:
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=youtube_url,
        add_video_info = False,
        language=["en", "en-US", "hi"]
    )
    docs = loader.load()
    return docs

#--2. Chunk the transcript--

def chunk_documents(docs: list, chunk_size: int = 1000, chunk_overlap = 150) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_documents(docs)

#--3. Build FAISS vector stores

def build_vectorstore(chunks: list)->FAISS:
    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs= {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True}
    )
    return FAISS.from_documents(chunks, embedding)

#--4. Format retrieved docs into a single string--

def format_docs(docs: list)->str:
    return "\n\n".join(doc.page_content for doc in docs)

# ──Format chat history as plain text (HF models don't take message objects)
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

#--5 Build the LCEL RAG system--

def build_rag_chain(vectorstore: FAISS, hf_api_token: str):
    retriever = vectorstore.as_retriever(          #create retriever
        search_type = "similarity",
        search_kwargs={"k":4}
    )

    endpoint = HuggingFaceEndpoint(
        repo_id= "deepseek-ai/DeepSeek-V4-Pro",
        huggingfacehub_api_token=hf_api_token,
        temperature = 0.2,
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
            context = lambda x: format_docs(retriever.invoke(x["standalone_question"]))
        ) 
        | answer_prompt | llm | parser
    )
    return {
        "condense_chain": condense_chain,
        "rag_chain": rag_chain,
        "retriever": retriever
    }

# ── 6. Ask a question (called per turn from app.py)--

def ask_question(chain_bundle: dict, question: str, chat_history: list) -> dict:
    """
    Args:
        chain_bundle : dict returned by build_rag_chain()
        question     : current user question (str)
        chat_history : list of LangChain message objects (HumanMessage / AIMessage)
 
    Returns:
        {"answer": str, "source_documents": list}
    """

    condense_chain = chain_bundle["condense_chain"]
    rag_chain = chain_bundle["rag_chain"]
    retriever = chain_bundle["retriever"]

    # Rephrase into standalone question if there is history
    if chat_history:
        standalone_question = condense_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        standalone_question = standalone_question.strip().splitlines()[0]
    else:
        standalone_question = question
    
    # Retrieve source docs for display in UI
    source_docs = retriever.invoke(standalone_question)

    # Generate the answer
    answer = rag_chain.invoke({
        "standalone_question": standalone_question,
        "question": question,
        "chat_history": chat_history
    })
    return {"answer": answer.strip(), "source_documents": source_docs}

# ── 7. One-shot convenience function ──

def process_youtube_url(youtube_url: str, hf_api_token: str) -> tuple:
    docs = load_transcript(youtube_url)
    chunks = chunk_documents(docs)
    vectorstores = build_vectorstore(chunks)
    chain_bundle = build_rag_chain(vectorstores, hf_api_token)
    metadata = docs[0].metadata if docs else {}
    return chain_bundle, metadata
