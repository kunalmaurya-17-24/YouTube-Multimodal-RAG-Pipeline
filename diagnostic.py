
import sys
import os
from dotenv import load_dotenv

print("--- Environment Check ---")
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")

try:
    import streamlit as st
    print("✅ Streamlit imported")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import cv2
    print("✅ OpenCV imported")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    from faster_whisper import WhisperModel
    print("✅ faster-whisper imported")
except ImportError as e:
    print(f"❌ faster-whisper import failed: {e}")

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    print("✅ NVIDIA NIM Endpoints imported")
except ImportError as e:
    print(f"❌ NVIDIA NIM import failed: {e}")

try:
    from langgraph.graph import StateGraph
    print("✅ LangGraph imported")
except ImportError as e:
    print(f"❌ LangGraph import failed: {e}")

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
nv_key = os.getenv("NVIDIA_API_KEY")

print("\n--- API Keys Check ---")
print(f"GROQ_API_KEY present: {'Yes' if groq_key else 'No'}")
print(f"NVIDIA_API_KEY present: {'Yes' if nv_key else 'No'}")
