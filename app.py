import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -------------------------------
# Config
# -------------------------------
FAISS_PATH = "vectorstore/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"


# -------------------------------
# Cache Heavy Resources
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300
    )


# -------------------------------
# Prompt
# -------------------------------
def build_prompt(context, question):
    return f"""
Answer the question strictly using the context below.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""


# -------------------------------
# RAG Logic
# -------------------------------
def answer_question(
    question,
    vectorstore,
    llm,
    top_k=2,
    max_chars=1500
):
    docs = vectorstore.similarity_search(question, k=top_k)

    context = "\n\n".join([doc.page_content for doc in docs])
    context = context[:max_chars]

    prompt = build_prompt(context, question)
    response = llm(prompt)[0]["generated_text"]

    return response, docs


# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="",
    layout="wide"
)

st.title("RAG Based Document Q&A System")
st.caption(
    "Ask questions over company policy and handbook documents using "
    "Retrieval Augmented Generation (RAG)."
)

st.markdown("---")


# -------------------------------
# Sidebar (Settings)
# -------------------------------
with st.sidebar:
    st.header("Settings")

    top_k = st.slider(
        "Number of retrieved sources",
        min_value=1,
        max_value=5,
        value=2
    )

    max_chars = st.slider(
        "Max context length",
        min_value=500,
        max_value=2000,
        value=1500,
        step=100
    )

    st.markdown("---")

    st.markdown("### How this works")
    st.markdown("""
    1. Your question is converted into an embedding  
    2. Similar document chunks are retrieved from FAISS  
    3. Retrieved text is injected into a prompt  
    4. The model answers **only** from that context  
    """)

    st.markdown("---")
    st.caption("Built with FAISS + Sentence Transformers + RAG")


# -------------------------------
# Load System
# -------------------------------
vectorstore = load_vectorstore()
llm = load_llm()


# -------------------------------
# Example Questions
# -------------------------------
st.markdown("#### Example Questions")
example_cols = st.columns(3)

examples = [
    "What happens if an employee resigns?",
    "Is leave encashment allowed?",
    "How to apply for parental leave?",
]

for col, example in zip(example_cols, examples):
    if col.button(example):
        st.session_state["question"] = example


# -------------------------------
# Question Input
# -------------------------------
question = st.text_input(
    "Ask a question about the documents:",
    value=st.session_state.get("question", "")
)


# -------------------------------
# Answer Section
# -------------------------------
if question:
    with st.spinner("Searching company policies..."):
        answer, docs = answer_question(
            question,
            vectorstore,
            llm,
            top_k=top_k,
            max_chars=max_chars
        )

    st.markdown("## Answer")
    st.write(answer)

  

    st.markdown("---")

    st.markdown("## Retrieved Source Chunks")
    for i, doc in enumerate(docs):
        with st.expander(f"Source {i + 1}"):
            st.write(doc.page_content)



