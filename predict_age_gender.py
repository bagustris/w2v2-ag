#!/usr/bin/env python3

import os
import audeer
import audonnx
import numpy as np
import audiofile
import audresample
import argparse

url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'

# make ~/models as root directory
root_dir = os.path.expanduser('~/models/w2v2-ag')

# create root directory if it does not exist
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)

cache_root = audeer.mkdir(os.path.join(root_dir, 'cache'))
model_root = audeer.mkdir(os.path.join(root_dir, 'model'))

archive_path = audeer.download_url(url, cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)
model = audonnx.load(model_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict age and gender from an audio file. Download the example of audio file from: https://huggingface.co/spaces/innnky/nene-emotion.')
    parser.add_argument('-i', '--input', type=str, 
                        default='./yachinene.wav', 
                        help='Path to input audio file')
    args = parser.parse_args()

    audio_file = args.input
    signal, sr = audiofile.read(audio_file)
    # convert to 16kHz if sr != 16kHz
    if sr != 16000:
        signal = audresample.resample(signal, sr, 16000)
    outs = model(signal, 16000)

    # print logits age converted to integer
    print(f"Age: {int(outs['logits_age'][0][0]*100)}")

    # for gender
    gender_logits = list(outs['logits_gender'][0])
    if np.argmax(gender_logits) == 0:
        gender = "female"
    elif np.argmax(gender_logits) == 1:
        gender = "male"
    else:
        gender = "child"    
    print(f"Gender or Child: {gender}")