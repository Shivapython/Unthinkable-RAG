import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime
import asyncio

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Handle event loop for async safety
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


# ----------- PDF PROCESSING -----------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# ----------- LLM CHAIN -----------
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "answer is not available in the context". 
    Don't provide wrong answers.
    
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# ----------- MAIN USER FUNCTION -----------
def user_input(user_question, api_key, pdf_docs, conversation_history):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return

    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    user_question_output = user_question
    response_output = response['output_text']
    pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []

    conversation_history.append(
        (user_question_output, response_output, "Google AI", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names))
    )

    # Display chat messages
    st.markdown(
        f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{ background-color: #2b313e; }}
            .chat-message.bot {{ background-color: #475063; }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{ width: 80%; padding: 0 1.5rem; color: #fff; }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question_output}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response_output}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show previous chat
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>    
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Download chat history
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    st.snow()


# ----------- MAIN APP -----------
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon="📚")
    st.header("Chat with multiple PDFs (v1) 📚")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    linkedin = "https://www.linkedin.com/in/ankitatiwary21/"
    kaggle = "https://www.kaggle.com/ankitatiwary21"
    github = "https://github.com/Ankitatiwary21"

    st.sidebar.markdown(
        f"[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin}) "
        f"[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]({kaggle}) "
        f"[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)]({github})"
    )

    api_key = st.secrets["api_keys"]["google"]

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done ✅")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""


if __name__ == "__main__":
    main()
