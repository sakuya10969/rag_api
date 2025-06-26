import os
from dotenv import load_dotenv

from langchain import hub
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from operator import itemgetter

# =======================
# 🔧 環境変数ロード
# =======================
load_dotenv()

AZ_OPENAI_ENDPOINT = os.getenv("AZ_OPENAI_ENDPOINT")
AZ_OPENAI_API_KEY = os.getenv("AZ_OPENAI_API_KEY")
DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZ_DOCUMENT_INTELLIGENCE_ENDPOINT")
DOC_INTELLIGENCE_KEY = os.getenv("AZ_DOCUMENT_INTELLIGENCE_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZ_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZ_SEARCH_ADMIN_KEY")
INDEX_NAME = "it_survey"

# =======================
# 📄 文書の読み込みと分割
# =======================
def load_and_split_document(file_path: str):
    loader = AzureAIDocumentIntelligenceLoader(
        file_path=file_path,
        api_key=DOC_INTELLIGENCE_KEY,
        api_endpoint=DOC_INTELLIGENCE_ENDPOINT,
        api_model="prebuilt-layout",
    )
    docs = loader.load()

    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    return text_splitter.split_text(docs[0].page_content)

# =======================
# 🧠 ベクトルストア初期化
# =======================
def init_vector_store(index_name: str):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
    )

    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    return vector_store

# =======================
# 🔎 RAGチェーン定義
# =======================
def create_rag_chain(retriever):
    prompt = hub.pull("rlm/rag-prompt")

    llm = AzureChatOpenAI(
        openai_api_version="2025-01-01-preview",
        azure_deployment="gpt-4o",
        temperature=0,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 通常のRAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ソース付きRAG
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = (
        RunnableMap({"documents": retriever, "question": RunnablePassthrough()})
        | {
            "documents": lambda input: [doc.metadata for doc in input["documents"]],
            "answer": rag_chain_from_docs,
        }
    )

    return rag_chain, rag_chain_with_source

# =======================
# 🚀 実行パート
# =======================
if __name__ == "__main__":
    splits = load_and_split_document("it_servey.pdf")

    vector_store = init_vector_store(INDEX_NAME)
    vector_store.add_documents(documents=splits)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    rag_chain, rag_chain_with_source = create_rag_chain(retriever)

    question = "この調査の目的は何ですか？"

    print("🔹 回答:")
    print(rag_chain.invoke(question))

    print("\n🔸 ソース付き:")
    print(rag_chain_with_source.invoke(question))
