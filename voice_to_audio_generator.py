import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename, duration=5, sr=22050):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')  # Record mono audio
    sd.wait()  # Wait for the recording to finish
    write(filename, sr, (audio * 32767).astype('int16'))  # Save audio as a WAV file
    print(f"Saved audio to {filename}")

# Example usage:
record_audio("reference_pronunciation.wav", duration=5)
record_audio("student_pronunciation.wav", duration=5)
