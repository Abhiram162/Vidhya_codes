import librosa
import numpy as np
from dtw import accelerated_dtw

def extract_mfcc(audio_path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose for time-major format

def compare_pronunciation(student_audio, reference_audio):
    student_mfcc = extract_mfcc(student_audio)
    reference_mfcc = extract_mfcc(reference_audio)
    
    # Perform DTW
    distance, _, _, _ = accelerated_dtw(student_mfcc, reference_mfcc, dist='euclidean')
    return distance

# student_audio = "student.wav"
# reference_audio = "reference.wav"
student_audio = "student_pronunciation.wav"
reference_audio = "reference_pronunciation.wav"
score = compare_pronunciation(student_audio, reference_audio)
if score < 10:
    feedback = "Excellent pronunciation!"
elif score < 20:
    feedback = "Good pronunciation, but needs slight improvement."
else:
    feedback = "Needs improvement. Practice more!"
    
print(f"Score: {score:.2f} - Feedback: {feedback}")