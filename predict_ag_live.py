#!/usr/bin/env python3
"""Live age and gender recognition from microphone signals."""
import argparse
import queue
import sys

import numpy as np
import sounddevice as sd

from predict_ag import predict_ag


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)

    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
        help='input channels to plot (default: the first)')
    parser.add_argument(
        '-d', '--device', type=int_or_str,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-w', '--window', type=float, default=200, metavar='DURATION',
        help='visible time slot (default: %(default)s ms)')
    parser.add_argument(
        '-i', '--interval', type=float, default=300,
        help='minimum time between plot updates (default: %(default)s ms)')
    parser.add_argument(
        '-b', '--blocksize', type=int, help='block size (in samples)')  
    parser.add_argument(
        '-sr', '--samplerate', type=float, default=16000, help='sampling rate of audio device')
    parser.add_argument(
        '-n', '--downsample', type=int, default=1, metavar='N',
        help='No downsample (default: %(default)s)')
    args = parser.parse_args(remaining)
    if any(c < 1 for c in args.channels):
        parser.error('argument CHANNEL: must be >= 1')
    mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
    q = queue.Queue()

    try:
        if args.samplerate is None:
            device_info = sd.query_devices(args.device, 'input')
            args.samplerate = device_info['default_samplerate']

        length = 3 * args.samplerate # 3 seconds
        plotdata = np.zeros((length, len(args.channels)))   # 3 seconds zeros

        stream = sd.InputStream(device=args.device, channels=max(args.channels),
                            samplerate=args.samplerate, callback=audio_callback)
        with stream:
            print("Listening... Press Ctrl+C to stop.")
            while True:
                try:
                    data = q.get()
                    # print(data)
                    # collect 3 seconds of data
                    if len(data) < length:
                        data = np.concatenate((plotdata[len(data):], data))
                        # predict age and gender from 5 
                    signal = data[:, 0].astype(np.float32)
                    print(f"Data of {signal.shape[0]/args.samplerate} seconds")
                    age, gender_prob = predict_ag(signal, args.samplerate)
                    gender = max(gender_prob, key=lambda k: gender_prob[k])
                    print(f"Age: {age}")
                    print(f"Gender: {gender}")
                except KeyboardInterrupt:
                    print("Stopped.")
                    break

    except Exception as e:
        print(f"Error: {type(e).__name__} - {str(e)}")
