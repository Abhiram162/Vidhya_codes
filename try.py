import librosa
import numpy as np
from dtw import accelerated_dtw
import matplotlib.pyplot as plt

def extract_mfcc(y, sr, n_mfcc=13, n_fft=1024):
    """
    Extract MFCC features from an audio signal.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
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

def time_stretch_bla(audio, target_len):
    """
    Time-stretch the audio to match a target length.
    """
    # Stretch factor is calculated as a ratio
    stretch_factor = float(len(audio)) / target_len
    return librosa.effects.time_stretch(audio, stretch_factor)  # Correctly apply time-stretch

def compare_all_segments(student_audio, reference_audio):
    # Step 1: Isolate active segments from both audio files
    student_segments, student_sr, _ = isolate_active_segments(student_audio)
    reference_segments, reference_sr, _ = isolate_active_segments(reference_audio)

    total_distance = 0

    # Step 2: Normalize the lengths of both segments (e.g., by time-stretching)
    for student_segment, reference_segment in zip(student_segments, reference_segments):
        # Time-stretch the shorter segment to match the length of the other
        if len(student_segment) == 0 or len(reference_segment) == 0:
            print("Skipping empty segment")
            continue

        # Time-stretch the student segment to match the reference length
        if len(student_segment) < len(reference_segment):
            student_segment = time_stretch_bla(student_segment, len(reference_segment))
        elif len(reference_segment) < len(student_segment):
            reference_segment = time_stretch_bla(reference_segment, len(student_segment))

        # Step 3: Extract MFCC features for the selected segments
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
