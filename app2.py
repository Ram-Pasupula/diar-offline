import torch
from pyannote.audio import Pipeline
from pyannote.audio import Model

import numpy as np
Model.from_pretrained("pyannote/segmentation-3.0",cache_dir="./pyannote")
diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", cache_dir="./pyannote")

if torch.cuda.is_available():
    diarize_pipeline = diarize_pipeline.to("cpu")

def diarize(audio_path, num_spk=None):
    
    diarization = diarize_pipeline(audio_path, num_speakers=num_spk)
    diarization_list = list()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_list.append([turn.start, turn.end, speaker])
    print(diarization_list)
    return diarization_list

diarize("./audio.wav",2)
