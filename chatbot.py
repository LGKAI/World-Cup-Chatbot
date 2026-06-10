from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def rag_chatbot():
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

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2}
    )

    template = (
        "You are a strict, citation-focused assistant for a private knowledge base.\n"
        "RULES:\n"
        "1) Use ONLY the provided context to answer.\n"
        "2) If the answer is not clearly contained in the context, say: "
        "\"I don't know based on the provided documents.\"\n"
        "3) Do NOT use outside knowledge, guessing, or web information.\n"
        "4) If applicable, cite sources as (source:page) using the metadata.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        user_input = input("Question: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        answer = rag_chain.invoke(user_input)
        print("Answer:", answer)

if __name__ == "__main__":
    rag_chatbot()