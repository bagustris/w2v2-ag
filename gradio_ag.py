#!/usr/bin/env python3
# gradio_ag.py: a python script that uses the gradio library to create a web
# app that allows users to predict age and gender based on given audio files.

import audiofile
import audresample
import gradio as gr

from predict_ag import predict_ag


def classify_speech(file):
    """Function that takes in an audio file and returns the predicted age and
    gender of the speaker."""
    audio, sr = audiofile.read(file)
    if sr != 16000:
        audio = audresample.resample(audio, sr, 16000)
    age, gender = predict_ag(audio, 16000)
    return age, gender


# audiofile_input = gr.inputs.Audio(type="filepath", label="Upload Audio File")

demo = gr.Interface(
    fn=classify_speech,
    inputs=gr.Audio(type="filepath", label="Upload Audio File"),
    # out 1 as age, output 2 as gender
    outputs=["text", "label"],
    examples=["./yachinene.wav"]
)

demo.launch()
