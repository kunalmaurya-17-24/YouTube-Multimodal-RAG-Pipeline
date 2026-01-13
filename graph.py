from typing import TypedDict, List, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
import operator
from processor import YouTubeProcessor
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
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
    # If chunks already exist in state, we might be skipping to RAG
    if state.get("chunks"):
        return state
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
    if state.get("error"): return state
    if state.get("chunks"): return state 
    
    print("---ANALYZING VISUALS WITH NVIDIA NIM (Phi-3 Vision)---")
    llm = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct")
    events = state.get("events", [])
    if not events:
        return {"error": "No events found to process"}

    processed_events = []
    # Transcription events are already in state
    processed_events.extend([e for e in events if e["type"] == "audio"])
    
    # Process frames with NVIDIA NIM
    visual_events = [e for e in events if e["type"] == "visual"]
    step = max(1, len(visual_events) // 10)
    selected_visuals = visual_events[::step][:10]
    
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    for event in selected_visuals:
        print(f"  NVIDIA NIM describing frame at {event['timestamp']:.2f}s...")
        base64_img = encode_image(event["frame_path"])
        msg = HumanMessage(content=[
            {"type": "text", "text": "Describe the visual content of this frame briefly."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
        ])
        try:
            res = llm.invoke([msg])
            event["description"] = res.content
            processed_events.append(event)
        except Exception as e:
            print(f"Vision Error at {event['timestamp']}s: {e}")
            # Continue with others if some fail, or return error if all fail
            continue

    # Re-sort events by timestamp
    processed_events.sort(key=lambda x: x['timestamp'])
    
    # Semantic Temporal Chunking with PROPER timestamp tracking
    chunks = []
    if not processed_events:
        return {"chunks": []}
    
    # Initialize first chunk with actual first event timestamp
    current_chunk = {
        "start": processed_events[0]['timestamp'], 
        "text": "", 
        "visual": ""
    }
    
    for event in processed_events:
        if event["type"] == "audio":
            current_chunk["text"] += " " + event["text"]
        else:
            current_chunk["visual"] += " " + event.get("description", "")
        
        # Create new chunk every 10 seconds
        if event["timestamp"] > current_chunk["start"] + 10:
            # Only add chunk if it has content
            if current_chunk["text"].strip() or current_chunk["visual"].strip():
                chunks.append(current_chunk)
            # Start new chunk with current event's timestamp
            current_chunk = {
                "start": event["timestamp"], 
                "text": "", 
                "visual": ""
            }
    
    # Add final chunk if it has content
    if current_chunk["text"].strip() or current_chunk["visual"].strip():
        chunks.append(current_chunk)
    
    return {"chunks": chunks}

def index_data(state: GraphState):
    if state.get("error"): return state
    if not state.get("chunks"): return state
    
    print("---INDEXING WITH NVIDIA NIM---")
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="adv_youtube_rag"
    )
    
    # Smart chunking to handle 512 token limit
    def chunk_text(text, max_words=400):
        """Split text into chunks with word count limit (~512 tokens)"""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunks.append(' '.join(words[i:i + max_words]))
        return chunks
    
    # Split large chunks before embedding
    texts = []
    metadatas = []
    for c in state["chunks"]:
        full_text = f"Time: {c['start']}s | Audio: {c['text']} | Visual: {c['visual']}"
        sub_chunks = chunk_text(full_text, max_words=400)
        
        for sub_chunk in sub_chunks:
            texts.append(sub_chunk)
            metadatas.append({"timestamp": c["start"], "type": "hybrid_chunk"})
    
    print(f"---Split {len(state['chunks'])} chunks into {len(texts)} sub-chunks for embedding---")
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return {"chunks": state["chunks"]} # Persist chunks for BM25 in RAG node

def rag_agent(state: GraphState):
    if state.get("error"): return state
    if not state.get("chunks") or len(state.get("chunks", [])) == 0:
        return {"error": "No indexed content found. Please index the video first."}

    print("---ADVANCED HYBRID RAG---")
    query = state.get("query", "Summarize this video.")
    
    # 1. Vector Search
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name="adv_youtube_rag")
    vector_results = vectorstore.similarity_search(query, k=5)
    
    # 2. BM25 Search (Keyword recall)
    chunks = state.get("chunks", [])
    tokenized_corpus = [f"{c.get('text', '')} {c.get('visual', '')}".split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_results = bm25.get_top_n(tokenized_query, chunks, n=3)
    
    # 3. Hybrid Context Construction
    context_parts = []
    for doc in vector_results:
        context_parts.append(f"[{doc.metadata.get('timestamp')}s]: {doc.page_content}")
    for chunk in bm25_results:
        context_parts.append(f"[{chunk['start']}s]: Audio: {chunk['text']} | Visual: {chunk['visual']}")
    
    context = "\n".join(list(set(context_parts))) # Simple dedup
    
    llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")
    prompt = f"""You are a precise video content assistant. Your job is to answer questions based ONLY on the provided video transcript and visual descriptions.

CRITICAL RULES:
1. ONLY use information explicitly stated in the Context below
2. If the question asks about something NOT in the Context, respond: "I don't have enough information about that in the video."
3. DO NOT add details, explanations, or general knowledge not in the Context
4. ALWAYS cite timestamps in format [MM:SS] or [XXXs] 
5. If the Context mentions the topic but doesn't fully answer the question, say so explicitly

Context from video:
{context}

User Question: {query}

Answer (grounded in Context only):
    """
    
    try:
        response = llm.invoke(prompt)
        return {"response": response.content}
    except Exception as e:
        return {"error": f"Connection Error (NVIDIA NIM): {str(e)}"}

def router(state: GraphState):
    if state.get("error"):
        return END
    if state.get("response"):
        return END
    # If we have query but no chunks/events, we need to process
    if state.get("query") and not state.get("chunks"):
        return "fetch_content"
    # if we have chunks and query, go to rag
    if state.get("chunks") and state.get("query"):
        return "rag_agent"
    return "fetch_content"

# Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("input_handler", input_handler)
workflow.add_node("fetch_content", fetch_content)
workflow.add_node("multimodal_processor", multimodal_processor)
workflow.add_node("index_data", index_data)
workflow.add_node("rag_agent", rag_agent)

workflow.set_entry_point("input_handler")
workflow.add_conditional_edges(
    "input_handler",
    router,
    {
        "fetch_content": "fetch_content",
        "rag_agent": "rag_agent",
        END: END
    }
)
workflow.add_edge("fetch_content", "multimodal_processor")
workflow.add_edge("multimodal_processor", "index_data")
workflow.add_edge("index_data", "rag_agent")
workflow.add_edge("rag_agent", END)

app = workflow.compile()
