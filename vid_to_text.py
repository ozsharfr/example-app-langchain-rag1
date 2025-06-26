import whisper
import yt_dlp

# Download audio from YouTube
video_url = "https://www.youtube.com/watch?v=kYINPLl_9rw"
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': 'audio.mp3',
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])

# Transcribe with Whisper
model = whisper.load_model("medium")  # 'medium' is a good balance between speed and accuracy for Hebrew
result = model.transcribe("audio.mp3", language="he")
print(result["text"])