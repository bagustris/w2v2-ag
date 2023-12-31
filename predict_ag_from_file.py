#!/usr/bin/env python3

import argparse

import audiofile
import audresample

from predict_ag import predict_ag

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
    age, gender_prob = predict_ag(signal, 16000)

    # get gender with highest probability
    gender = max(gender_prob, key=lambda k: gender_prob[k])
    # print logits age converted to integer
    print(f"Age: {age}")    
    print(f"Gender: {gender}")
