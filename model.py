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

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Based on: https://github.com/ufal/npfl114/blob/master/labs/03/uppercase.py

parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=None, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=None, type=int, help="Batch size.")
parser.add_argument("--epochs", default=None, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=None, type=int, help="Window size to use.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout")
parser.add_argument("--clean", default=False, help="Clean data", action='store_true')
parser.add_argument("--layers", default="256,128,64", help="Dense layers")
parser.add_argument("--experiment_id", default="X", type=str, help="Experiment id")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--lr_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--decay", default=None, type=str, help="Learning rate decay")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing")

def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    for k, v in sorted(vars(args).items()):
        print(f"{k}={v}")

    # Create logdir name
    args.logdir = os.path.join("logs", f"{args.experiment_id}")

    # Load data
    torlakian_data = TorlakianData(args.window, args.alphabet_size, args.clean)

    print("-" * 30)
    print(f"Experiment {args.experiment_id}")
    
    layers_dims = args.layers.split(",")
    layers_dims = [int(l) for l in layers_dims]

    layers = [
        tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32),
        tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(torlakian_data.train.alphabet))),
        tf.keras.layers.Flatten()
    ]

    for layer_dim in layers_dims:
        layers.append(tf.keras.layers.Dense(layer_dim, activation=tf.nn.relu))
        layers.append(tf.keras.layers.Dropout(args.dropout))

    layers.append(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model = tf.keras.Sequential(layers)

    learning_rate = args.lr
    if args.decay is not None:
        decay_steps = torlakian_data.train.size/ args.batch_size * args.epochs
        if args.decay == 'linear':
            learning_rate = tf.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=args.lr, 
                decay_steps=decay_steps, 
                end_learning_rate=args.lr_final
            )
        elif args.decay == 'exponential':
            decay_rate = args.lr_final / args.lr
            learning_rate = tf.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.lr,
                decay_steps=decay_steps,
                decay_rate=decay_rate
            )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[tf.metrics.BinaryAccuracy(name="accuracy")],
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=100, profile_batch=0)

    logs = model.fit(
        torlakian_data.train.data["windows"], 
        torlakian_data.train.data["labels"],
        batch_size=args.batch_size, 
        epochs=args.epochs,
        validation_data=(torlakian_data.val.data["windows"], torlakian_data.val.data["labels"]),
        callbacks=[tb_callback, stop_early],
    )

    def do_uppercase(index):
        letter = torlakian_data.test.text[index]
        if output[index] and letter in torlakian_data.test.alphabet:
            return letter.upper()
        else:
            return letter

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "torlak_test.txt"), "w", encoding="utf-8") as predictions_file:
        start_time = time.time()
        
        output = model(torlakian_data.test.data["windows"])
        
        # The actiation function in the output layer was sigmoid, 
        # so by doing this operation we get boolean values on the output
        # that state if the current character should be uppercased or not
        output = output > 0.5

        # Transforming to uppercase to mark the accent position
        # is done with the function `do_uppercase`
        predictions = map(do_uppercase, range(len(output)))
        output_text = ''.join(predictions)
        predictions_file.write(output_text)

        end_time = time.time()
        print(f"Prediction took: {end_time - start_time} s")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)