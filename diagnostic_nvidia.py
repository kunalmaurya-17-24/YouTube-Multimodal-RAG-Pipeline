import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

def test_chat():
    print("\n--- Testing NVIDIA NIM Chat (Llama 3.1) ---")
    try:
        llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct", api_key=api_key)
        res = llm.invoke("Say 'Network OK'")
        print(f"✅ Success: {res.content}")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_embeddings():
    print("\n--- Testing NVIDIA NIM Embeddings ---")
    try:
        embed = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", api_key=api_key)
        res = embed.embed_query("Hello world")
        print(f"✅ Success: Embedded vector of length {len(res)}")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_vision():
    print("\n--- Testing NVIDIA NIM Vision (Phi-3) ---")
    try:
        llm = ChatNVIDIA(model="microsoft/phi-3-vision-128k-instruct", api_key=api_key)
        # Testing with a dummy message (no image to simplify)
        res = llm.invoke("Describe this image: [DUMMY]")
        print(f"✅ Success: {res.content[:50]}...")
    except Exception as e:
        print(f"❌ Failed: {e}")

def test_transcription_endpoint():
    print("\n--- Testing NVIDIA NIM ASR Endpoint Reachability ---")
    url = "https://integrate.api.nvidia.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        # Simple GET or empty POST to check reachability
        res = requests.get("https://integrate.api.nvidia.com/v1/models", headers=headers)
        if res.status_code == 200:
            print("✅ Success: integrate.api.nvidia.com is reachable and API key is valid.")
        else:
            print(f"❌ Failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    if not api_key:
        print("❌ Error: NVIDIA_API_KEY not found in .env")
    else:
        test_chat()
        test_embeddings()
        test_vision()
        test_transcription_endpoint()
