import os
import json
import streamlit as st
from dotenv import load_dotenv
import torch
import tempfile
from langchain_groq import ChatGroq
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.document_loaders import UnstructuredPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.messages import HumanMessage, AIMessage

# Constants
##MODEL_NAME = "google/flan-t5-xxl"
MODEL_NAME = "google/flan-t5-base"
VECTOR_STORE_DIR = "vector_store_chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load environment variables
load_dotenv()
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

if huggingface_api_token is None:
    st.error("Please set the HUGGINGFACE_TOKEN environment variable.")
    st.stop()

def load_file(file):
    if file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        st.write("Loading PDF file...")
        #loader = UnstructuredPDFLoader(tmp_file_path)
        loader = PyPDFDirectoryLoader(tmp_file_path)
        documents = loader.load()
    return documents

def split_text(documents):
    st.write("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_updated_vector_store(texts, embeddings):
    st.write("Creating vector store...")
    vector_store = Chroma.from_documents(texts, embeddings, persist_directory=VECTOR_STORE_DIR)
    vector_store.persist()
    return vector_store

def model_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe

def get_chat_history(session_id) -> BaseChatMessageHistory:
    path = os.path.join("chat_histories", f"{session_id}.json")
    os.makedirs("chat_histories", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("[]")
    return FileChatMessageHistory(file_path=path)

def track_uploaded_files(session_id, filename):
    os.makedirs("uploaded_files", exist_ok=True)
    path = os.path.join("uploaded_files", f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            files = json.load(f)
    else:
        files = []
    if filename not in files:
        files.append(filename)
        with open(path, "w") as f:
            json.dump(files, f, indent=2)

os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectordb = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

#llm = HuggingFacePipeline(pipeline=model_inference(), model_kwargs={"temperature": 0.7})
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and user's latest question, generate a query to retrieve relevant documents from the vector store."),
    MessagesPlaceholder("chat_history"),
    ("human", "User's question: {input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=retriever_prompt
)

llm_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a resume AI Assistant. Given the user's question and the retrieved documents use them to answer the question, generate a detailed answer to the question. Also refer the chat history if needed."),
    ("human", "User's question: {input}"),
    MessagesPlaceholder("chat_history"),
    ("human", "Retrieved documents: {context}")
])

document_chain = create_stuff_documents_chain(llm=llm, prompt=llm_prompt)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=document_chain
)

chat_with_history = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.title("LangChain + Hugging Face Resume Parser")
st.write("Upload PDF resumes and ask questions.")
session_id = st.text_input("Session ID", value="default_session")

if "session_id" not in st.session_state:
    st.session_state.session_id = session_id

upload_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
all_texts = []

if upload_files:
    for file in upload_files:
        if file is not None:
            documents = load_file(file)
            texts = split_text(documents)
            all_texts.extend(texts)
            track_uploaded_files(session_id, file.name)
    if all_texts:
        vector_store = create_updated_vector_store(all_texts, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.success("Files loaded and processed.")

st.divider()

user_query = st.text_input("Ask a question about the resumes:", key="user_query")

if user_query and session_id:
    with st.spinner("Generating answer..."):
        response = chat_with_history.invoke(input={"input": user_query}, config={"configurable": {"session_id": session_id}})
        chat_with_history.get_session_history(session_id).add_user_message(user_query)
        chat_with_history.get_session_history(session_id).add_ai_message(response["answer"] if isinstance(response, dict) and "answer" in response else str(response))
        st.write("Answer:", response["answer"])
                 
        st.write("whole response:", response)

st.sidebar.subheader("Uploaded Files")
uploaded_path = os.path.join("uploaded_files", f"{session_id}.json")
if os.path.exists(uploaded_path):
    with open(uploaded_path) as f:
        files = json.load(f)
        for file in files:
            st.sidebar.write(f"- {file}")
else:
    st.sidebar.write("No files uploaded.")

st.sidebar.subheader("Conversation History")
chat_path = os.path.join("chat_histories", f"{session_id}.json")
if os.path.exists(chat_path):
    with open(chat_path) as f:
        messages = json.load(f)
        for msg in messages:
            role = msg["type"]
            content = msg["data"]["content"]
            st.sidebar.markdown(f"**{role.capitalize()}**: {content}")
else:
    st.sidebar.write("No conversation yet.")
