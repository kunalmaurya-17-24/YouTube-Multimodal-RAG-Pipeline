import os
import yt_dlp
import cv2
from typing import Dict, Any, List, Optional
import shutil

class YouTubeProcessor:
    def __init__(self, download_dir: str = "./downloads"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def fetch_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch video metadata using yt-dlp."""
        ydl_opts = {'quiet': True, 'extract_flat': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)

    def process_multimodal(self, url: str, interval_seconds: int = 5) -> List[Dict[str, Any]]:
        """
        Download video/audio, transcribe with timestamps, and extract frames.
        Returns a list of 'events' with timestamps.
        """
        # 1. Download best audio
        print(f"---STARTING AUDIO DOWNLOAD: {url}---")
        temp_audio_path = os.path.join(self.download_dir, "audio_raw")
        audio_path = os.path.join(self.download_dir, "audio.wav")
        if os.path.exists(audio_path): os.remove(audio_path)
        
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': temp_audio_path,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([url])
            
        print("---CONVERTING TO WAV (16kHz Mono) for RIVA gRPC---")
        # Find the actual downloaded file (yt-dlp adds extension)
        downloaded_file = None
        for f in os.listdir(self.download_dir):
            if f.startswith("audio_raw"):
                downloaded_file = os.path.join(self.download_dir, f)
                break
        
        if downloaded_file:
            os.system(f"ffmpeg -i \"{downloaded_file}\" -ar 16000 -ac 1 -c:a pcm_s16le \"{audio_path}\" -y")
            os.remove(downloaded_file)
        else:
            raise Exception("Audio download failed - no file found")

        # 2. Transcribe with NVIDIA gRPC (Official API Implementation)
        print("---TRANSCRIBING WITH NVIDIA gRPC (Riva Whisper)...---")
        print("---Step 1: Importing gRPC modules...---")
        
        import grpc
        from riva.client.proto import riva_asr_pb2, riva_asr_pb2_grpc, riva_audio_pb2
        
        transcript_events = []
        channel = None
        
        try:
            print("---Step 2: Creating secure gRPC channel...---")
            # Create channel with optimized settings
            channel = grpc.secure_channel(
                'grpc.nvcf.nvidia.com:443',
                grpc.ssl_channel_credentials(),
                options=[
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 30000),
                    ('grpc.keepalive_timeout_ms', 10000),
                ]
            )
            
            print("---Step 3: Creating RivaSpeechRecognition stub...---")
            stub = riva_asr_pb2_grpc.RivaSpeechRecognitionStub(channel)
            
            print("---Step 4: Loading audio file...---")
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            print(f"---Step 5: Loaded {len(audio_bytes)/1024/1024:.1f}MB of audio data---")
            
            print("---Step 6: Building RecognitionConfig...---")
            # Build config according to official API documentation
            config = riva_asr_pb2.RecognitionConfig(
                encoding=riva_audio_pb2.AudioEncoding.LINEAR_PCM,  # From riva_audio.proto
                sample_rate_hertz=16000,
                language_code="en-US",
                max_alternatives=1,
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            )
            
            print("---Step 7: Building RecognizeRequest...---")
            # Build request according to official API documentation
            request = riva_asr_pb2.RecognizeRequest(
                config=config,
                audio=audio_bytes
            )
            
            print("---Step 8: Preparing NVCF metadata...---")
            # NVCF authentication metadata
            metadata = (
                ('function-id', 'b702f636-f60c-4a3d-a6f4-f3568c13bd7d'),
                ('authorization', f'Bearer {os.getenv("NVIDIA_API_KEY")}')
            )
            
            print("---Step 9: Calling Recognize method (this may take a few minutes)...---")
            # Call the Recognize method with metadata
            response = stub.Recognize(request, metadata=metadata, timeout=600)
            
            print("---Step 10: Processing transcription response...---")
            
            # Parse response according to official API documentation
            for result in response.results:
                if not result.alternatives:
                    continue
                    
                alt = result.alternatives[0]
                
                if alt.words:
                    # Group words into ~10s segments for better RAG context
                    start_time = alt.words[0].start_time / 1000.0  # Convert ms to seconds
                    text_acc = ""
                    
                    for word in alt.words:
                        text_acc += word.word + " "
                        end_time = word.end_time / 1000.0  # Convert ms to seconds
                        
                        # Create segment if we've accumulated 10+ seconds
                        if end_time - start_time > 10.0:
                            if text_acc.strip():
                                transcript_events.append({
                                    "timestamp": (start_time + end_time) / 2,
                                    "text": text_acc.strip(),
                                    "type": "audio",
                                    "start": start_time,
                                    "end": end_time
                                })
                            start_time = end_time
                            text_acc = ""
                    
                    # Add remaining text
                    if text_acc.strip():
                        transcript_events.append({
                            "timestamp": (start_time + end_time) / 2,
                            "text": text_acc.strip(),
                            "type": "audio",
                            "start": start_time,
                            "end": end_time
                        })
                else:
                    # Fallback: use full transcript without word timings
                    if alt.transcript.strip():
                        transcript_events.append({
                            "timestamp": 0,
                            "text": alt.transcript.strip(),
                            "type": "audio",
                            "start": 0,
                            "end": 0
                        })
            
            if not transcript_events:
                raise Exception("No transcription results returned from NVIDIA")
            
            print(f"---Step 11: Successfully extracted {len(transcript_events)} transcript segments---")
                
        except grpc.RpcError as e:
            error_details = f"Code: {e.code()}, Details: {e.details()}"
            print(f"---ERROR: gRPC call failed - {error_details}---")
            raise Exception(f"Transcription Failed: {error_details}")
        except Exception as e:
            print(f"---ERROR: {str(e)}---")
            raise Exception(f"Transcription Failed: {str(e)}")
        finally:
            # Always close the channel
            if channel:
                print("---Closing gRPC channel...---")
                channel.close()
        
        print(f"---COMPLETED TRANSCRIPTION: {len(transcript_events)} segments.---")

        # 3. Extract Frames with timestamps
        print("---STARTING VIDEO DOWNLOAD FOR FRAMES---")
        video_path = os.path.join(self.download_dir, "temp_video.mp4")
        if os.path.exists(video_path): os.remove(video_path)
        
        ydl_opts_video = {
            'format': 'bestvideo[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': video_path,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            ydl.download([url])

        print("---STARTING FRAME EXTRACTION (OpenCV)...---")
        frames_dir = os.path.join(self.download_dir, "frames")
        if os.path.exists(frames_dir): shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)
        
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30 # Fallback
        
        frame_interval = int(fps * interval_seconds)
        
        success, image = vidcap.read()
        count = 0
        visual_events = []
        
        while success:
            if count % frame_interval == 0:
                seconds = count / fps
                frame_name = f"frame_{int(seconds)}.jpg"
                frame_path = os.path.join(frames_dir, frame_name)
                cv2.imwrite(frame_path, image)
                visual_events.append({
                    "timestamp": seconds,
                    "frame_path": frame_path,
                    "type": "visual"
                })
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        
        print(f"---COMPLETED: Extracted {len(visual_events)} frames and {len(transcript_events)} audio segments.---")
        
        # Cleanup video but keep frames for now
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(audio_path): os.remove(audio_path)

        return sorted(transcript_events + visual_events, key=lambda x: x['timestamp'])

    def cleanup(self):
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)
