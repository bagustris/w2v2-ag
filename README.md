# Predict Age and Gender from Audio Files  
This is a Python CLI wrappers for Audeering public age and gender model (Original repo: https://github.com/audeering/w2v2-age-gender-how-to).
This script uses a pre-trained ONNX model to predict the age and gender of a person from an audio file and live microphone.

## Installation

To use this script, you need to have Python 3 installed on your system. You also need to install the following Python packages:

- `numpy`
- `audiofile`
- `audresample`
- `audeer`
- `audonnx`

More details about these packages can be found in the `requirements.txt` file.

You can install these packages using `pip -r requirements.txt` under (Mini)conda environment:

```bash
$ conda create -n w2v2-ag python=3.9
$ conda activate w2v2-ag
$ pip install -r requirements.txt

```

## Usage

To use the script, run the following command in your terminal:

```
$ python predict_ag_from_file.py [-h] [-i INPUT]
```

The script takes the following arguments:

- `-h`, `--help`: Show the help message and exit.
- `-i INPUT`, `--input INPUT`: Path to input audio file. Default is `./yachinene.wav`.

The script outputs the predicted age and gender of the person in the audio file.

Here's an example usage of the script:

```bash
# predicting age and gender from file
$ python predict_ag_from_file.py -i input.wav
Age: 21
Gender or Child: female
# predicting age and gender from live microphone
$ python predict_ag_live.py
```

The first will predict the age and gender of the person in the `input.wav` file.
If you don't specify an input file, the script will use the default file `~/wav/yachinene.wav`.

The latter command will predict the age and gender of the live microphone.

## License

This script is licensed under the MIT License. See the `LICENSE` file for more information.
