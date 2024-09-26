import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

st.title("Scientific Paper Search App")

api_key_option = st.radio("Choose API key:", ("OpenAI", "Course"))

if api_key_option == "OpenAI":
    open_api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if open_api_key:
        os.environ["OPENAI_API_KEY"] = open_api_key
    else:
        st.warning("Please enter an OpenAI API key to proceed.")
        st.stop()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    embeddings_api_model = OpenAIEmbeddings(model="text-embedding-3-small")

else:
    course_api_key = st.text_input("Enter your LLM course API key:", type="password")
    if course_api_key:
        pass
    else:
        st.warning("Please enter a LLM course API key to proceed.")
        st.stop()

    from utils import ChatOpenAI, OpenAIEmbeddings

    llm = ChatOpenAI(temperature=0.7, course_api_key=course_api_key)
    embeddings_api_model = OpenAIEmbeddings(course_api_key=course_api_key)

documents = []

data_source = st.radio("Choose data source:", ("Upload PDFs", "Scan 'data' folder"))

if data_source == "Upload PDFs":
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if not os.path.exists("data"):
            os.makedirs("data")

        for uploaded_file in uploaded_files:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        for uploaded_file in uploaded_files:
            loader = PyPDFLoader(os.path.join("data", uploaded_file.name))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name  # Add filename to metadata
            documents.extend(docs)
else:
    if os.path.exists("data"):
        for filename in os.listdir("data"):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join("data", filename))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = filename  # Add filename to metadata
                documents.extend(docs)

        if not documents:
            st.write("No PDF files found in the 'data' folder.")
    else:
        st.write("The 'data' folder does not exist.")

if documents:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    split_documents = splitter.split_documents(documents)

    db = FAISS.from_documents(split_documents, embeddings_api_model)
    db.save_local("faiss_db")

    retriever = db.as_retriever(
        search_type="similarity",
        k=10,
        score_threshold=None,
    )

    def generate_response(question, retriever, llm):
        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are an assistant for writing scientific papers.
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Answer:
        """

        response = llm.invoke(prompt).content

        used_articles = set([d.metadata["source"] for d in docs])

        final_result = f"{response}\n\nArticles used for context:\n\n"
        final_result += "\n".join(used_articles)

        return final_result

    user_question = st.text_input("Enter your question about the papers:")

    if user_question:
        result = generate_response(user_question, retriever, llm)
        st.write(result)
else:
    st.write("Please upload PDF files or ensure the 'data' folder contains PDF files to proceed.")
