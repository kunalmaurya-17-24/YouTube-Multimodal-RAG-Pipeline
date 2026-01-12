import streamlit as st
from dotenv import load_dotenv
import os
from graph import app as langgraph_app

load_dotenv()

def main():
    st.set_page_config(page_title="YouTube Multimodal RAG", layout="wide")
    st.title("📺 YouTube Multimodal RAG Pipeline")
    st.markdown("---")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        google_key = st.text_input("Google AI Key (for transcription sanity check)", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        nvidia_key = st.text_input("NVIDIA API Key (for Embeddings)", type="password", value=os.getenv("NVIDIA_API_KEY", ""))
        
        if api_key: os.environ["GROQ_API_KEY"] = api_key
        if google_key: os.environ["GOOGLE_API_KEY"] = google_key
        if nvidia_key: os.environ["NVIDIA_API_KEY"] = nvidia_key

    # Main Input
    youtube_url = st.text_input("Enter YouTube Video URL")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Index Video (Advanced RAG)"):
            if not youtube_url:
                st.error("Please enter a URL.")
            elif not os.getenv("NVIDIA_API_KEY") or not os.getenv("GROQ_API_KEY"):
                st.error("Missing API Keys (Groq/NVIDIA).")
            else:
                with st.status("🏗️ Building Video Index...", expanded=True) as status:
                    st.write("Initializing Multimodal Agent...")
                    inputs = {"url": youtube_url}
                    for output in langgraph_app.stream(inputs):
                        for key, value in output.items():
                            st.write(f"Node `{key}` completed.")
                    status.update(label="✅ Index Ready!", state="complete", expanded=False)
                st.success("Temporal chunks indexed in ChromaDB with NVIDIA NIM!")

    with col2:
        st.subheader("Temporal Chat")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask about content at a specific time..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            with st.spinner("Retrieving from Video Index..."):
                # We need the chunks from the state to perform BM25
                # Since we don't persist state across app runs easily without a DB, 
                # we'll run the RAG part of the graph.
                result = langgraph_app.invoke({"url": youtube_url, "query": user_query})
                response = result.get("response", "I don't have enough information.")
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
