from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wav

# Path to the audio file
audio_file = "./audio.wav"

# Perform speaker diarization
#segments = aS.speaker_diarization(audio_file,2)  # 3 represents the number of clusters (speakers)

# # Print the speaker segments
# for seg in segments:
#     if isinstance(seg, (list, tuple)) and len(seg) >= 3:
#         print("Speaker:", seg[0], " - Segment:", seg[1], "-", seg[2])
#     else:
#         print("Invalid segment format:", seg)

# Perform diarization
# Perform diarization
# Read the audio file to get the sampling rate and data
sampling_rate, data = wav.read(audio_file)

# Define parameters for diarization
st_win = 0.05  # Short-term window size in seconds
st_step = 0.025  # Short-term window step in seconds

# Perform diarization
segments = aS.silence_removal(data, sampling_rate, st_win, st_step)

# Print the timestamps and speaker labels
for i, segment in enumerate(segments):
    print("Speaker", i+1, ":", segment[0], "-", segment[1])