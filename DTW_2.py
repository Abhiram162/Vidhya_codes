import librosa
import numpy as np
from dtw import accelerated_dtw
import matplotlib.pyplot as plt

def extract_mfcc(y, sr, n_mfcc=13):
    """
    Extract MFCC features from an audio signal.
    """
    n_fft = min(2048, len(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,n_fft=n_fft)
    return mfcc.T  # Transpose to time-major format

def isolate_active_segments(audio_path, sr=22050, top_db=20):
    """
    Isolate active segments in an audio file based on energy levels.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    intervals = librosa.effects.split(y, top_db=top_db)  # Detect active regions
    active_segments = [y[start:end] for start, end in intervals]
    return active_segments, sr, intervals

def calculate_distance_matrix(mfcc1, mfcc2):
    """
    Calculate a distance matrix between two MFCC feature sets using DTW.
    """
    distance, cost_matrix, acc_cost_matrix, path = accelerated_dtw(mfcc1, mfcc2, dist="euclidean")
    return distance, cost_matrix, path

def plot_distance_matrix(cost_matrix, path):
    """
    Visualize the distance matrix and alignment path.
    """
    plt.imshow(cost_matrix.T, origin="lower", cmap="viridis", aspect="auto")
    plt.plot(path[0], path[1], "w")  # Alignment path
    plt.colorbar()
    plt.title("Distance Matrix with Alignment Path")
    plt.xlabel("Input MFCC Frames")
    plt.ylabel("Reference MFCC Frames")
    plt.show()

def compare_all_segments(student_audio, reference_audio):
    # Step 1: Isolate active segments from both audio files
    student_segments, student_sr, _ = isolate_active_segments(student_audio)
    reference_segments, reference_sr, _ = isolate_active_segments(reference_audio)

    total_distance = 0

    # Step 2: Compare all segments from both audio files
    for student_segment, reference_segment in zip(student_segments, reference_segments):
        student_mfcc = extract_mfcc(student_segment, student_sr)
        reference_mfcc = extract_mfcc(reference_segment, reference_sr)

        # Perform DTW on the current segments
        distance, _, _ = calculate_distance_matrix(student_mfcc, reference_mfcc)
        total_distance += distance  # Accumulate the distance

    # Output the total distance
    print(f"Total DTW distance across all segments: {total_distance}")


# Example usage:
student_audio = "student_pronunciation.wav"
reference_audio = "reference_pronunciation.wav"
compare_all_segments(student_audio, reference_audio)
