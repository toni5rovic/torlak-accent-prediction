#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
import time
from torlakian_data import TorlakianData

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Based on: https://github.com/ufal/npfl114/blob/master/labs/03/uppercase.py

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model.h5", type=str, help="Model weights")
parser.add_argument("text", default="", type=str, help="Text or file to process")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    WINDOW = 8
    DROPOUT = 0.5
    ALPHABET_SIZE = 38
    WINDOW = 8
    CLEAN = True
    
    # Load data
    torlakian_data = TorlakianData(WINDOW, ALPHABET_SIZE, CLEAN)

    layers = [
        tf.keras.layers.Input(shape=[2 * WINDOW + 1], dtype=tf.int32),
        tf.keras.layers.Lambda(lambda x: tf.one_hot(x, ALPHABET_SIZE)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ]

    model = tf.keras.Sequential(layers)

    model.load_weights(args.model)
    
    def do_uppercase(index):
        letter = dataset.text[index]
        if output[index] and letter in dataset.alphabet:
            return letter.upper()
        else:
            return letter

    input_text = ""
    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
            input_text = ''.join(lines)
    else:
        input_text = args.text

    dataset = torlakian_data.Dataset(input_text, WINDOW, ALPHABET_SIZE, CLEAN)
    output = model(dataset.data["windows"])
    output = output > 0.5

    predictions = map(do_uppercase, range(len(output)))
    output_text = ''.join(predictions)
    
    print(output_text)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)