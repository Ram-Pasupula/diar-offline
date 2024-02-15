import librosa
import numpy as np
from sklearn.cluster import KMeans

# Path to the audio file
audio_file = "audio.wav"

# Load the audio file
audio, sr = librosa.load(audio_file)

# Extract features (e.g., MFCCs)
mfccs = librosa.feature.mfcc(y=audio, sr=sr)

# Transpose MFCCs to have features along rows
mfccs = mfccs.T

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)  # Specify the number of clusters (speakers)
labels = kmeans.fit_predict(mfccs)
# Initialize an empty list to store segment start and end timestamps
timestamps = []
# Initialize a dictionary to store segments grouped by speaker
segments_by_speaker = {label: [] for label in set(labels)}
# Define the duration of each frame in seconds
frame_duration = len(audio) / len(mfccs)
# Iterate over each segment and assign start and end timestamps
# Iterate over each segment and assign start and end timestamps
for i, label in enumerate(labels):
    start_time = i * frame_duration
    end_time = (i + 1) * frame_duration
    timestamps.append((start_time, end_time, label))
    segments_by_speaker[label].append((start_time, end_time))

# Print the speaker segments with timestamps
for speaker, segments in segments_by_speaker.items():
    print(f"Speaker {speaker}:")
    for i, (start_time, end_time) in enumerate(segments):
        print(f"Segment {i}: Start: {start_time:.2f} s, End: {end_time:.2f} s")
