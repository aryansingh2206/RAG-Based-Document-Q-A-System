import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


DATA_PATH = "data/documents"
FAISS_PATH = "vectorstore/faiss_index"


def load_documents():
    """
    Load all PDF documents from the data folder.
    """
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

            print(f"Loaded {len(docs)} pages from {file}")

    return documents


def chunk_documents(documents):
    """
    Split documents into overlapping chunks for better semantic retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    return chunks


def generate_embeddings(chunks):
    """
    Generate embeddings separately (for validation / understanding).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings


from langchain_huggingface import HuggingFaceEmbeddings

def store_in_faiss(chunks, path=FAISS_PATH):
    texts = [chunk.page_content for chunk in chunks]

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_function
    )

    vectorstore.save_local(path)
    return vectorstore



if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()
    print(f"\nTotal pages loaded: {len(documents)}")

    print("\nChunking documents...")
    chunks = chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    print("\nGenerating embeddings (validation step)...")
    embeddings = generate_embeddings(chunks)
    print("Embedding shape:", embeddings.shape)

    print("\nStoring chunks in FAISS...")
    store_in_faiss(chunks)

    print("\nFAISS index created and saved successfully 🚀")
