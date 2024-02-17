# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 libsndfile1 ffmpeg

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
#RUN pip install pyannote.audio==3.1 pyannote.core pyannote.pipeline pyannote.algorithms pyannote.database pyannote.metrics pyannote.parser

RUN pip install --no-cache-dir -r requirements.txt

ENV PYANNOTE_CACHE="./pyannote"
ENV TORCH_HOME="./pyannote"

EXPOSE 8000
# Run app.py when the container launches
CMD ["uvicorn", "app:app","--host","0.0.0.0","--port","8000"]
