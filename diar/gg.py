from google.cloud import speech_v1p1beta1 as speech

# Path to the audio file
audio_file = "audio.wav"

# Initialize the client
client = speech.SpeechClient()

# Configure the audio input
audio = speech.types.RecognitionAudio(uri=audio_file)

# Configure the speaker diarization
config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    language_code='en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count=2  # Specify the expected number of speakers
)

# Perform the transcription
response = client.recognize(config=config, audio=audio)

# Print the result
for result in response.results:
    print('Speaker:', result.channel_tag)
    print('Transcript:', result.alternatives[0].transcript)
