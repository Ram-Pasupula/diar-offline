import nemo.collections.asr as nemo_asr
import numpy as np
from IPython.display import Audio, display
import librosa
import os
import wget
import matplotlib.pyplot as plt

import nemo
import glob

print("Hi")
ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
if not os.path.exists(os.path.join(data_dir,'an4_diarize_test.wav')):
    AUDIO_FILENAME = wget.download(an4_audio_url, data_dir)
else:
    AUDIO_FILENAME = os.path.join(data_dir,'an4_diarize_test.wav')

audio_file_list = glob.glob(f"{data_dir}/*.wav")
print("Input audio file list: \n", audio_file_list)

signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)
print(signal)
print(sample_rate)

from omegaconf import OmegaConf
import shutil
DOMAIN_TYPE = "meeting" # Can be meeting or telephonic based on domain type of the audio file
CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

if not os.path.exists(os.path.join(data_dir,CONFIG_FILE_NAME)):
    CONFIG = wget.download(CONFIG_URL, data_dir)
else:
    CONFIG = os.path.join(data_dir,CONFIG_FILE_NAME)

cfg = OmegaConf.load(CONFIG)
#print(OmegaConf.to_yaml(cfg))

# Create a manifest file for input with below format. 
# {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-", 
# "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}
import json
meta = {
    'audio_filepath': AUDIO_FILENAME, 
    'offset': 0, 
    'duration':None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': None, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}
with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')
#print(cfg.diarizer.manifest_filepath)

pretrained_speaker_model='titanet_large'
cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
cfg.diarizer.out_dir = data_dir #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.clustering.parameters.oracle_num_speakers=False


# Using Neural VAD and Conformer ASR 
cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
cfg.diarizer.asr.model_path = '/Users/rampasupula/Downloads/stt_en_conformer_ctc_large.nemo'
#cfg.diarizer.asr.model_path ='openai/whisper-tiny.en'
cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
cfg.diarizer.asr.parameters.asr_based_vad = False
#Run ASR and get word timestamps
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
asr_model = asr_decoder_ts.set_asr_model()
word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

print("Decoded word output dictionary: \n", word_hyp['an4_diarize_test'])
print("Word-level timestamps dictionary: \n", word_ts_hyp['an4_diarize_test'])

# from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
# asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
#asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
#Run diarization with the extracted word timestamps

# diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
#print("Diarization hypothesis output: \n", diar_hyp['an4_diarize_test'])