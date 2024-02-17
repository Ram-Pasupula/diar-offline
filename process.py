import os
from transformers import pipeline

from pyannote.audio import Pipeline
from io import StringIO
import torch

from diarize_process import DiarizationPipeline

os.environ["TORCH_HOME"] = "./pyannote"
os.environ["HF_HOME"] = "./pyannote"
os.environ["PYANNOTE_CACHE"] = "./pyannote"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", cache_dir="./pyannote"
)
pipeline.to(torch.device(device))

pipeline = DiarizationPipeline(diarization_pipeline=pipeline)


def diarization(audio):
    segments = pipeline(audio)
    # output_file = StringIO()
    # print(segments, file=output_file, flush=True)
    # output_file.seek(0)
    return segments
