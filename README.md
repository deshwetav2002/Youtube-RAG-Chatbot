# Youtube-RAG-Chatbot
1. YouTube Video Q&A — Conversational RAG System  |  Python · LangChain · HuggingFace · FAISS
2. Built an end-to-end conversational RAG pipeline that ingests YouTube video transcripts and answers user 
questions strictly grounded in video content, preventing hallucination.
3. Implemented a two-chain architecture: a condense chain to rephrase follow-up questions into standalone 
queries, and a RAG chain to retrieve relevant transcript chunks and generate accurate answers.
4. Used HuggingFaceEmbeddings and FAISS vectorstore for semantic chunk retrieval, and LangChain's 
RecursiveCharacterTextSplitter for optimal document chunking.
5. Maintained full multi-turn conversational context using LangChain ChatPromptTemplate, 
MessagesPlaceholder, and HumanMessage/AIMessage history tracking
