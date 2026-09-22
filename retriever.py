from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def load_params(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def load_documents_from_directory(directory_path: str) -> list:
    loader = DirectoryLoader(directory_path, glob='**/*.txt', loader_cls=TextLoader)
    documents = loader.load()
    return documents

def create_vector_store(docs: list, embeddings) -> Chroma:
    """Creating a Chroma vector store from the documents and embeddings."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 100, chunk_overlap=25)
    docs_splits = text_splitter.split_documents(docs)
    # Create a Chroma vector store from the documents
    db = Chroma.from_documents(
        documents=docs_splits, 
        embedding=embeddings,
        collection_name="sourcechat",
        )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    return retriever




def retriever():
    # Load parameters from params.yaml
    params = load_params('params.yaml')
    embedding_model_name = params['model']['embedding']
    #llm_model_name = params['model']['llm']

    print(f"Using embedding model: {embedding_model_name}")

    documents = load_documents_from_directory('data')
    print(f"Loaded {len(documents)} documents from the 'data' directory.")


    # Initialize the embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model_name)

    retriever = create_vector_store(documents, embedding_model)

    return retriever
    