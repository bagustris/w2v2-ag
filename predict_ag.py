#!/usr/bin/env python3

import os

import audeer
import audonnx
import numpy as np


def predict_ag(signal, sr) -> tuple:
    """Function to predict age and gender from an audio signal.
    Args:
        signal: audio signal (sr = 16kHz)
    Returns:
        age: age prediction (integer)
        gender or child: female, male, or child (string)
    Examples:
        >>> import audiofile
        >>> signal = audiofile.read('yachinene.wav')
        >>> predict_ag(signal)
        Age: 46
        Gender or Child: male
    """
    # url to download the model
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

    # assure sr = 16kHz, if not raise error
    assert sr == 16000, "Sampling rate must be 16kHz."
    outs = model(signal, 16000)

    # print logits age converted to integer
    age = int(outs['logits_age'][0][0]*100)
    # print(f"Age: {age}")

    # for gender, male and female only
    gender_logits = list(outs['logits_gender'][0][:-1])
    
    # output dictionary of {female: probility, male: probability}
    gender = {'female': float(gender_logits[0]), 'male': float(gender_logits[1])}
    # if np.argmax(gender_logits) == 0:
    #     gender = "female"
    # else:
    #     gender = "male"
    # else:
    #     gender = "child"    
    # # print(f"Gender or Child: {gender}")

    return age, gender
