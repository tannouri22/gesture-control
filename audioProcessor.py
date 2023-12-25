import pyaudio
import wave
import tempfile
import os
import whisper
import keyboard
import time
import numpy as np

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

def audioChunk():
    audio = pyaudio.PyAudio()
    # Set up the audio stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=44100,
                        input=True,
                        frames_per_buffer=1024)
    startTime = time.time()
    frames = []
    while True:
        duration = time.time() - startTime
        if duration < 4:
            data = stream.read(1024)
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

def deleteTempfile(temp_filename):
    # Delete the temporary file
    try:
        os.remove(temp_filename)
        print(f"Temporary file {temp_filename} deleted.")
    except OSError as e:
        print(f"Error deleting temporary file: {e}")

def rollingAudio(filePath, audioToggle):
    while True:
        audioFrames = audioChunk()
        tempFile = save_audio_to_tempfile(audioFrames)
        filePath.put(tempFile.encode('utf-8'))
        if audioToggle.value == 0:
            audioToggle.value = 1

def transcribeAudio(filePath, audioText):
    while True:
        if filePath.empty() == False:
            audioPath = filePath.get().decode('utf-8')
            # # Pass the numpy array to the transcribe method
            result = model.transcribe(audioPath)    
            print(result['text'])
            audioText.value = result['text'].encode('utf-8')
            deleteTempfile(audioPath)
        else:
            print("No audio file found.")

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
    deleteTempfile(temp_file_path)