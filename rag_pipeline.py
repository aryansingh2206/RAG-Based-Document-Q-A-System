from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


FAISS_PATH = "vectorstore/faiss_index"


# -----------------------------------
# Load FAISS Vector Store
# -----------------------------------
def load_vectorstore():
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# -----------------------------------
# Load Lightweight Free LLM (CPU Safe)
# -----------------------------------
def load_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300
    )

    return llm_pipeline


# -----------------------------------
# Hallucination-Safe Prompt
# -----------------------------------
def build_prompt(context, question):
    prompt = f"""
Answer the question strictly using the context below.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt


# -----------------------------------
# RAG Query Logic
# -----------------------------------
def answer_question(question, vectorstore, llm, k=2, max_chars=1500):
    docs = vectorstore.similarity_search(question, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])
    context = context[:max_chars]  # prevent token overflow

    prompt = build_prompt(context, question)
    response = llm(prompt)[0]["generated_text"]

    return response, docs


# -----------------------------------
# CLI Test Interface
# -----------------------------------
if __name__ == "__main__":
    print("🔄 Loading vector store...")
    vectorstore = load_vectorstore()

    print("🤖 Loading LLM...")
    llm = load_llm()

    print("\n✅ RAG system ready. Ask a question!\n")

    while True:
        question = input("❓ Your question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer, retrieved_docs = answer_question(question, vectorstore, llm)

        print("\n📌 Answer:\n")
        print(answer)

        print("\n📄 Retrieved Context Chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n--- Chunk {i + 1} ---")
            print(doc.page_content[:300])
