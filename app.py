import streamlit as st
import os
from rag_system import RAGChatbot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Rag chatbot with Qwen",
    layout="wide"
)

if "chatbot" not in st.session_state:
    st.session_state.chatbot=None
if "messages" not in st.session_state:
    st.session_state.messages=[]
if "loading" not in st.session_state:
    st.session_state.loading=False

def initialize_chatbot():
    try:
        with st.spinner("Loading Qwen model and processing documents..."):
            st.session_state.chatbot = RAGChatbot()
        st.success("Chatbot initialized Successfully")
        return True
    except Exception as e:
        st.error(f"Error Initializing Chatbot : {str(e)}")
        logger.error(f"Chatbot Initialization error : {str(e)}")
        return False
    
def main():
    st.title("Rag chatbot with Qwen")
    st.markdown("---")

    with st.sidebar:
        st.header("Document Management")

        uploaded_file=st.file_uploader(
            "Upload a document",
            type=['pdf','txt'],
            help="Upload PDF or TEXT files to add to the knowledge base"
        )
        if uploaded_file is not None:
            upload_dir="documents"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file_path= os.path.join(upload_dir,uploaded_file.name)
            with open(file_path,"wb") as f:
                f.write(uploaded_file.getbuffer())

            if st.session_state.chatbot:
                if st.session_state.chatbot.add_document(file_path):
                    st.success(f"Added {uploaded_file.name} to knowledge base")
                else:
                    st.error(f"Failed to add {uploaded_file.name}")
            else:
                st.info("File saved. Initialize chatbot to process it.")
            
        st.markdown("---")

        if st.session_state.chatbot:
            st.header("Knowledge Based Stats")
            stats=st.session_state.chatbot.vector_store.get_stats()
            st.metric("Total Documents",stats['total_documents'])
            st.metric("Vector Dimensions",stats['dimension'])
        st.markdown("---")

        if st.button("Clear Conversation"):
            st.session_state.messages=[]
            st.rerun()

        if st.session_state.chatbot is None:
            if st.button("Initialize Chatbot"):
                if initialize_chatbot():
                    st.rerun()
        else:
            st.success("Chatbot Ready")
            if st.button("Reinitialize"):
                st.session_state.chatbot=None
                st.rerun()


    if st.session_state.chatbot is None:
        st.info("Click 'Initialize Chatbot' in the sidebar to get started")
        st.markdown("""
        ### Welcome to RAG Chatbot!
        
        1. **Initialize the chatbot** using the sidebar button
        2. **Upload documents** (PDF or TXT) to build your knowledge base
        3. **Ask questions** about your documents
        """)
        return


    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if message['role']=='assistant' and "context" in message:
                st.markdown(message['content'])
                expander=st.expander("Retrieved Context")
                with expander:
                    st.write(message['context'])
            else:
                st.markdown(message['content'])


    if prompt := st.chat_input("Ask a question about your documents..."):

        st.session_state.messages.append({'role':"user","content":prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
        

        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chatbot.chat(prompt)

                    if result['status']=='success':
                        st.markdown(result['response'])

                        st.session_state.messages.append({
                            "role":"assistant",
                            "content":result['response'],
                            "context":result['context']
                        })
                        expander=st.expander("Retrieved Context")
                        with expander:
                            st.write(result['context'])
                    else:
                        error_message=result['response']
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role":"assistant",
                            "content":error_message
                        })
                except Exception as e:
                    error_msg=f"Sorry, I encountered an error : {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role":"assistant",
                        "content":error_msg
                    })

if __name__=='__main__':
    main()