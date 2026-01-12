from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
import operator
from processor import YouTubeProcessor
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from rank_bm25 import BM25Okapi
import os
import base64
import json

class GraphState(TypedDict):
    url: str
    query: str
    metadata: Dict[str, Any]
    events: List[Dict[str, Any]] # Combined transcript + visual events
    chunks: List[Dict[str, Any]] # Timestamp-aligned semantic chunks
    response: str
    error: str

def input_handler(state: GraphState):
    if not state.get("url"):
        return {"error": "Missing URL"}
    return state

def fetch_content(state: GraphState):
    print("---FETCHING METADATA & MEDIA---")
    processor = YouTubeProcessor()
    try:
        metadata = processor.fetch_metadata(state["url"])
        # process_multimodal handles download, transcription, and frame extraction
        events = processor.process_multimodal(state["url"])
        return {"metadata": metadata, "events": events}
    except Exception as e:
        return {"error": str(e)}

def multimodal_processor(state: GraphState):
    print("---DESCRIBING VISUALS & ALIGNING CHUNKS---")
    llm = ChatGroq(model="llama-3.2-11b-vision-preview")
    events = state["events"]
    
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    processed_events = []
    for event in events:
        if event["type"] == "visual":
            print(f"  Describing frame at {event['timestamp']:.2f}s...")
            base64_img = encode_image(event["frame_path"])
            msg = HumanMessage(content=[
                {"type": "text", "text": "Describe the visual content of this frame briefly."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ])
            res = llm.invoke([msg])
            event["description"] = res.content
        processed_events.append(event)
    
    # Semantic Temporal Chunking: Aligning transcript and visuals by timestamp
    # For simplicity: creating chunks every 10 seconds of content
    chunks = []
    current_chunk = {"start": 0, "text": "", "visual": ""}
    for event in processed_events:
        if event["type"] == "audio":
            current_chunk["text"] += " " + event["text"]
        else:
            current_chunk["visual"] += " " + event["description"]
        
        # Every 10s or at end, finalize chunk
        if event["timestamp"] > current_chunk["start"] + 10:
            chunks.append(current_chunk)
            current_chunk = {"start": event["timestamp"], "text": "", "visual": ""}
    chunks.append(current_chunk)
    
    return {"chunks": chunks}

def index_data(state: GraphState):
    print("---INDEXING WITH NVIDIA NIM---")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="adv_youtube_rag"
    )
    
    texts = [f"Time: {c['start']}s | Audio: {c['text']} | Visual: {c['visual']}" for c in state["chunks"]]
    metadatas = [{"timestamp": c["start"], "type": "hybrid_chunk"} for c in state["chunks"]]
    
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return {"chunks": state["chunks"]} # Persist chunks for BM25 in RAG node

def rag_agent(state: GraphState):
    print("---ADVANCED HYBRID RAG---")
    query = state.get("query", "Summarize this video.")
    
    # 1. Vector Search
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="adv_youtube_rag")
    vector_results = vectorstore.similarity_search(query, k=5)
    
    # 2. BM25 Search (Keyword recall)
    tokenized_corpus = [f"{c['text']} {c['visual']}".split() for c in state["chunks"]]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_results = bm25.get_top_n(tokenized_query, state["chunks"], n=3)
    
    # 3. Hybrid Context Construction
    context_parts = []
    for doc in vector_results:
        context_parts.append(f"[{doc.metadata.get('timestamp')}s]: {doc.page_content}")
    for chunk in bm25_results:
        context_parts.append(f"[{chunk['start']}s]: Audio: {chunk['text']} | Visual: {chunk['visual']}")
    
    context = "\n".join(list(set(context_parts))) # Simple dedup
    
    llm = ChatGroq(model="llama3-70b-8192")
    prompt = f"""
    You are a professional video analysis assistant. 
    Using the STRICT context provided below, answer the user query.
    
    RULES:
    1. Always cite the timestamp in brackets like [MM:SS] or [SSs].
    2. If the answer is not in the context, say "I don't have enough information based on the video."
    3. Do NOT hallucinate details not mentioned in the audio or visual descriptions.
    
    Context:
    {context}
    
    User Query: {query}
    """
    
    response = llm.invoke(prompt)
    return {"response": response.content}

# Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("input_handler", input_handler)
workflow.add_node("fetch_content", fetch_content)
workflow.add_node("multimodal_processor", multimodal_processor)
workflow.add_node("index_data", index_data)
workflow.add_node("rag_agent", rag_agent)

workflow.set_entry_point("input_handler")
workflow.add_edge("input_handler", "fetch_content")
workflow.add_edge("fetch_content", "multimodal_processor")
workflow.add_edge("multimodal_processor", "index_data")
workflow.add_edge("index_data", "rag_agent")
workflow.add_edge("rag_agent", END)

app = workflow.compile()
