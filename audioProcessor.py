import pyaudio
import wave
import tempfile
import os
import whisper
import keyboard

model = whisper.load_model('small.en')
buttonClicked = False

def record_audio(sample_rate=44100, channels=2, chunk_size=1024):
    audio = pyaudio.PyAudio()

    # Set up the audio stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Press Enter to start recording...")
    keyboard.wait("/")

    print("Recording... Press Enter again to stop.")

    frames = []
    while True:
        if keyboard.is_pressed("/"):
            data = stream.read(chunk_size)
            frames.append(data)
        else:
            break

    print("Recording finished.")

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return frames

def save_audio_to_tempfile(frames, sample_width=2, sample_rate=44100, channels=2):
    # Create a temporary file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name

        # Set up the wave file parameters
        wave_file = wave.open(temp_filename, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()

    return temp_filename

def delete_tempfile(temp_filename):
    # Delete the temporary file
    try:
        os.remove(temp_filename)
        print(f"Temporary file {temp_filename} deleted.")
    except OSError as e:
        print(f"Error deleting temporary file: {e}")

if __name__ == "__main__":
    # Record audio
    audio_frames = record_audio()

    # Save audio to a temporary file
    temp_file_path = save_audio_to_tempfile(audio_frames)
    
    print(f"Audio saved to: {temp_file_path}")

    #process audio
    result = model.transcribe(temp_file_path)
    print(result['text'])

    # Delete the temporary file
    delete_tempfile(temp_file_path)