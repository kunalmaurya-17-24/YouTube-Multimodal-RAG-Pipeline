import os
import yt_dlp
import cv2
from typing import Dict, Any, List, Optional
from faster_whisper import WhisperModel
import shutil

class YouTubeProcessor:
    def __init__(self, download_dir: str = "./downloads"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        # Initialize whisper model once
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    def fetch_metadata(self, url: str) -> Dict[str, Any]:
        """Fetch video/playlist metadata using yt-dlp."""
        ydl_opts = {'quiet': True, 'extract_flat': 'in_playlist'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)

    def process_multimodal(self, url: str, interval_seconds: int = 5) -> List[Dict[str, Any]]:
        """
        Download video/audio, transcribe with timestamps, and extract frames.
        Returns a list of 'events' with timestamps.
        """
        # 1. Download best audio
        audio_path = os.path.join(self.download_dir, "audio.mp3")
        ydl_opts_audio = {
            'format': 'bestaudio/best',
            'outtmpl': audio_path.replace(".mp3", ""),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
            ydl.download([url])

        # 2. Transcribe with timestamps
        segments, _ = self.whisper_model.transcribe(audio_path, beam_size=5)
        transcript_events = []
        for segment in segments:
            transcript_events.append({
                "timestamp": (segment.start + segment.end) / 2,
                "text": segment.text.strip(),
                "type": "audio",
                "start": segment.start,
                "end": segment.end
            })

        # 3. Extract Frames with timestamps
        video_path = os.path.join(self.download_dir, "temp_video.mp4")
        ydl_opts_video = {
            'format': 'bestvideo[height<=720][ext=mp4]/best[ext=mp4]/best',
            'outtmpl': video_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            ydl.download([url])

        frames_dir = os.path.join(self.download_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
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
        
        # Cleanup video but keep frames for now
        if os.path.exists(video_path): os.remove(video_path)
        if os.path.exists(audio_path): os.remove(audio_path)

        return sorted(transcript_events + visual_events, key=lambda x: x['timestamp'])

    def cleanup(self):
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)
