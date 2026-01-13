import streamlit as st
from dotenv import load_dotenv
import os
from graph import app as langgraph_app

load_dotenv()

def main():
    st.set_page_config(page_title="YouTube Multimodal RAG", layout="wide")
    st.title("📺 YouTube Multimodal RAG Pipeline")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Only show inputs if keys are missing
        nv_ready = os.getenv("NVIDIA_API_KEY") is not None
        
        if not nv_ready:
            nv_key = st.text_input("NVIDIA API Key", type="password")
            if nv_key: os.environ["NVIDIA_API_KEY"] = nv_key
        else:
            st.sidebar.success("✅ NVIDIA Key loaded!")

    # Main Input
    youtube_url = st.text_input("Enter YouTube Video URL")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Index Video (Advanced RAG)"):
            if not youtube_url:
                st.error("Please enter a URL.")
            elif not os.getenv("NVIDIA_API_KEY"):
                st.error("Missing NVIDIA API Key. Please check Sidebar or .env")
            else:
                with st.status("🚀 Processing Video...", expanded=True) as status:
                    progress_bar = st.progress(0, text="Initializing...")
                    
                    # Map node names to friendly messages
                    node_messages = {
                        "input_handler": "🔍 Validating URL...",
                        "fetch_content": "📥 Downloading Media...",
                        "multimodal_processor": "🧠 Analyzing with NVIDIA NIM...",
                        "index_data": "🗂️ Creating Temporal Index..."
                    }
                    
                    inputs = {"url": youtube_url}
                    final_state = {}
                    nodes = list(node_messages.keys())
                    
                    for output in langgraph_app.stream(inputs):
                        for key, value in output.items():
                            msg = node_messages.get(key, f"Processing {key}...")
                            st.write(msg)
                            
                            # Calculate progress percentage
                            if key in nodes:
                                progress = (nodes.index(key) + 1) / len(nodes)
                                progress_bar.progress(progress, text=msg)
                            
                            final_state.update(value)
                    
                    if "chunks" in final_state:
                        st.session_state.chunks = final_state["chunks"]
                    
                    if final_state.get("error"):
                        status.update(label="❌ Indexing Failed", state="error", expanded=True)
                        st.error(f"Pipeline Error: {final_state['error']}")
                    else:
                        progress_bar.progress(1.0, text="✅ Processing Complete!")
                        status.update(label="✅ Video Indexed!", state="complete", expanded=False)
                        st.balloons()
                        st.success("🎉 Video processed! You can now ask any question below.")

    with col2:
        st.subheader("Temporal Chat")
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask about content at a specific time..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)
            
            with st.spinner("Retrieving from Video Index..."):
                # Pass existing chunks to the graph to skip indexing nodes
                inputs = {
                    "url": youtube_url, 
                    "query": user_query, 
                    "chunks": st.session_state.chunks
                }
                result = langgraph_app.invoke(inputs)
                
                if result.get("error"):
                    response = f"❌ Error: {result['error']}"
                else:
                    response = result.get("response", "I don't have enough information.")
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
