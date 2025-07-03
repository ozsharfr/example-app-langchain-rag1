import whisper
import ffmpeg

def transcribe_hebrew_video(video_path, output_path=None):
    """
    Transcribe Hebrew video to text using Whisper
    
    Args:
        video_path: Path to video file
        output_path: Optional path to save transcript (default: same name as video with .txt extension)
    """
    
    # Load Whisper model (medium or large work best for Hebrew)
    model = whisper.load_model("large")
    
    # Extract audio from video (Whisper can handle video directly, but audio extraction is more reliable)
    audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Transcribe with Hebrew language specified
        result = model.transcribe(audio_path, language='he', fp16=False)
        
        # Get the transcript
        transcript = result["text"]
        
        # Save to file if output path specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
        else:
            # Default output file
            default_output = video_path.rsplit('.', 1)[0] + '_transcript.txt'
            with open(default_output, 'w', encoding='utf-8') as f:
                f.write(transcript)
        
        print(f"Transcription completed. Text saved to: {output_path or default_output}")
        return transcript
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    
    finally:
        # Clean up temporary audio file
        import os
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Usage example
if __name__ == "__main__":
    video_file = "1000027317.mp4"
    transcript = transcribe_hebrew_video(video_file)
    if transcript:
        print("\nTranscript preview:")
        print(transcript[::-1])