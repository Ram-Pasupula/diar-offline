from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import glob
import time
import os
import logging
from process import diarization
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

app = FastAPI(
    title="Speaker-diarization",
    openapi_url="/gpt/api/v1/models/speaker-diarization/api/openapi.json",
    docs_url="/gpt/api/v1/models/speaker-diarization/",
    redoc_url="/gpt/api/v1/models/speaker-diarization/redoc",
    generate_schema=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("Diarization process")

@app.get("/")
async def health():
    return "OK"

@app.post("/diarize")
async def diarize(file: UploadFile = File(...)
):
    try:
        start_time = time.time()
        file_content = await file.read()
        result = diarization(file_content)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        logger.info(f"Execution time: {elapsed_time} seconds")
    except Exception as error: 
        logger.error("An exception thrown while diarize:", exec_info=True)
        raise Exception("An exception thrown while diarize:", type(error).__name__,"-",error)
    else:
        return result
    finally:
        logger.info("Successfully diarized")
