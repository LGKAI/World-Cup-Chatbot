import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

def get_vectorstore_and_splits(embeddings):
    index_path = "faiss_index"
    splits_path = "splits.pkl"
    if os.path.exists(index_path) and os.path.exists(splits_path):
        print("Loading existing FAISS index and BM25 splits...")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore, splits
    
    print("No existing index found. Loading documents and building FAISS & BM25 index...")
    loader = DirectoryLoader(
        path="./papers",
        glob="**/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
        is_separator_regex=True
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    print("Saving FAISS index locally...")
    vectorstore.save_local(index_path)
    
    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)
        
    return vectorstore, splits

def rag_chatbot():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore, splits = get_vectorstore_and_splits(embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 5
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4]
    )

    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

    # 1. History Aware Retriever Prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is. "
        "The reformulated question MUST be in Vietnamese."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, ensemble_retriever, contextualize_q_prompt
    )

    # 2. QA Prompt
    qa_system_prompt = (
        "You are a helpful and engaging AI assistant answering questions based solely on the provided context.\n"
        "RULES:\n"
        "1) You MUST respond entirely in VIETNAMESE.\n"
        "2) Give the direct answer first, then provide some interesting details, related stats, or context from the provided documents to make the answer more engaging.\n"
        "3) Do NOT use robotic introductory phrases like 'Theo tài liệu được cung cấp...' or 'Dựa vào ngữ cảnh...'. Make it sound natural and conversational.\n"
        "4) Use ONLY the provided context. If the answer is not in the context, simply say: \"Tôi không tìm thấy thông tin này trong tài liệu.\"\n"
        "5) Do NOT use outside knowledge.\n"
        "6) Be extremely accurate with numbers, tables, and data comparisons.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # 3. Chains
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []
    
    print("\nChatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
            
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        answer = result["answer"]
        print("Answer:", answer)
        
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=answer)
        ])

if __name__ == "__main__":
    rag_chatbot()