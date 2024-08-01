import streamlit as st
from src.helper import get_pdf_text, get_text_chunk, get_vector_store, get_conversational_chain

def user_input():
    user_question = st.session_state.user_question_input
    if user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chatHistory = response['chat_history']
        
        # Clear the input field after processing
        st.session_state.user_question_input = ""

def main():
    st.set_page_config('File Crawl')
    st.header("File Crawl Bot")
    st.subheader('Get the information from PDF files.')

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if 'chatHistory' not in st.session_state:
        st.session_state.chatHistory = []

    with st.sidebar:
        st.image("crawl.png", caption="File Crawl", width=100, channels='RGB')
        st.title("Menu")
        pdf_doc = st.file_uploader(
            'Upload your multiple PDF Files and click on submit & process button.',
            accept_multiple_files=True,
            type=["pdf"]  # Specify allowed file types
        )
        if st.button('Submit & Process'):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_doc)
                text_chunks = get_text_chunk(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Success")

    # Disable the input field if conversation is not initialized
    input_disabled = st.session_state.conversation is None

    # Use columns to keep input at a fixed position
    col1, col2 = st.columns([3, 1])

    with col1:
        # Create the text input for user questions with a callback
        st.text_input(
            "Ask Question from the PDF Files",
            key="user_question_input",
            disabled=input_disabled,
            on_change=user_input  # Callback to process input when changed
        )
        st.text("Please upload files in the sidebar and press submit in order")
        st.text("to get started and ask questions about documents")
        
        # Display chat history below the input
        if st.session_state.chatHistory:
            for i, message in enumerate(st.session_state.chatHistory):
                if i % 2 == 0:
                    st.markdown(f"**User:** {message.content}")
                else:
                    st.markdown(f"**Reply:** {message.content}")

if __name__ == '__main__':
    main()
