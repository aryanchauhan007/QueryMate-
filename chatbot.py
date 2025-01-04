import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


st.set_page_config(page_title="🤖 QueryMate", page_icon=":scroll")
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text(pdf_docs):
    """Extract text from uploaded PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_coversational_chain():
    """Define the conversational chain with custom prompts."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the
    provided context, say "answer is not available in the context." Do not guess or provide false information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process user input and generate a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5) 
    #st.write("Retrieved Documents for Debugging:", docs)  # Debugging( optional for retrivial check of information from PDFs.)

    chain = get_coversational_chain()

    try:
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")

def main():
    """Main function to handle the app logic."""
    st.header("Multi-PDF's 📚- QueryMate 🤖")
    st.write("Upload your PDF files, and ask any question from their content!")

   
    user_question = st.text_input("Ask your query from the uploaded PDF(s)... ✍️📝")

    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.image("img/freepik__candid-image-photography-natural-textures-highly-r__17115.jpeg")
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & Click on the Submit & Process Button", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                raw_text = get_text(pdf_docs)
                st.write("Extracted text successfully!")  # Debugging
                text_chunks = get_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Your PDFs have been processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

st.write("---")
st.write("An AI App implemented by @Aryan Chauhan")

# Footer
st.markdown(
    """
     <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
        © <a href="https://github.com/aryanchauhan007" target="_blank">Aryan Chauhan</a> | Made with ❤️
     </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()
