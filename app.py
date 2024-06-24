import os
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceInstructEmbeddings



def save_uploaded_file(uploaded_file, directory):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_faiss_index(folder_path, index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    faiss_index = FAISS.load_local(folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
    return faiss_index

def load_metadata(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        metadata = pickle.load(f)
    return metadata

def load_csv_data(csv_file_path):
    csv_data = pd.read_csv(csv_file_path)
    video_data = dict(zip(csv_data["Video_title"], csv_data["Video_link"]))
    return video_data

def get_conversation_chain(vectorstore):
    api_key = os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", token=api_key)
    conversation_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='similarity'),
        return_source_documents=True
    )
    return conversation_chain

def handle_userinput(user_question, video_data):
    response = st.session_state.conversation({'question': user_question})
    answer = response['answer']
    sources = response['source_documents']
    
    st.markdown(f"**Question:** {user_question}")
    st.markdown(f"**Answer:** {answer}")
    
    st.markdown("**Relevant webinars from IAEA.org:**")
    source_urls = []
    source_urls = set()  # Use a set to track unique URLs
    for source in sources:
        video_title = source.metadata['source']
        video_link = video_data.get(video_title, "")
        if video_link not in source_urls:  # Check if the URL is already in the set
            source_urls.add(video_link)  # Add the URL to the set
            st.markdown(f"{video_link}")

def main():
    #load_dotenv()
    # Custom HTML/CSS for the banner
    custom_html = """
                <div style="display: flex; justify-content: center; overflow: hidden; height: 200px; background-color: #f0f0f0;">
                    <img src="https://raw.githubusercontent.com/HelgeSverre/mistral/a762dae337e959cc8e51c868eca9a406977c6407/art/header.png" alt="Banner Image" style="height: 200px;">
                </div>
                """
    # Display the banner
    st.markdown(custom_html, unsafe_allow_html=True)

    st.title("Mistral AI Chat Application")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_question = st.chat_input("Ask a question about IAEA webinars:")
    if user_question:
        handle_userinput(user_question, st.session_state.video_data)

    with st.sidebar:
        st.subheader("Load FAISS Index and CSV Data")
        index_files = st.file_uploader("Upload your FAISS index and CSV files here:", type=["faiss", "pkl", "csv"], accept_multiple_files=True)
        if st.button("Load Data"):
            if index_files and len(index_files) == 3:
                with st.spinner("Loading FAISS index and CSV data"):
                    index_path = ""
                    pkl_path = ""
                    csv_path = ""
                    for uploaded_file in index_files:
                        if uploaded_file.name.endswith(".faiss"):
                            index_path = save_uploaded_file(uploaded_file, "index_files")
                        elif uploaded_file.name.endswith(".pkl"):
                            pkl_path = save_uploaded_file(uploaded_file, "index_files")
                        elif uploaded_file.name.endswith(".csv"):
                            csv_path = save_uploaded_file(uploaded_file, "index_files")

                    if index_path and pkl_path and csv_path:
                        index_folder_path = os.path.dirname(index_path)
                        index_file = os.path.splitext(os.path.basename(index_path))[0]
                        faiss_index = load_faiss_index(index_folder_path, index_file)
                        metadata = load_metadata(pkl_path)
                        video_data = load_csv_data(csv_path)
                        st.session_state.conversation = get_conversation_chain(faiss_index)
                        st.session_state.metadata = metadata  # Save metadata for later use
                        st.session_state.video_data = video_data  # Save video data for later use
                        st.success("FAISS index and CSV data loaded successfully!")
                    else:
                        st.error("Please upload a .faiss, .pkl, and .csv file.")

if __name__ == '__main__':
    main()
