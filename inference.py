#!/usr/bin/env python3
from typing import List
from torlakian_data import TorlakianData
import tensorflow as tf

def do_uppercase(output: tf.Tensor, dataset: TorlakianData.Dataset) -> str:
    """Does uppercasing on the text from `dataset` on the positions where `ouput` is `True`"""

    def uppercase_one_char(index):
        letter, should_uppercase = output_and_text[index]
        if should_uppercase and letter in dataset.alphabet:
            return letter.upper()
        else:
            return letter

    # `dataset.text` and `output` are of the same length because the model
    # outputs one boolean value for each character in the original text.
    output_and_text = list(zip(dataset.text, output))
    predictions = map(uppercase_one_char, range(len(output_and_text)))
    return ''.join(predictions)

def inference(model: tf.keras.Sequential, dataset: TorlakianData.Dataset) -> str:
    """Does inference using `model` on the data from `dataset` and 
    returns text which is uppercased in positions where model predicts it should be uppercased."""
    raw_output = model(dataset.data["windows"])
    
    # The activation function in the output layer was sigmoid, 
    # so by doing this operation we get boolean values on the output
    # that state if the current character should be uppercased or not
    output = raw_output > 0.5

    # Transforming to uppercase to mark the accent position
    # is done with the function `do_uppercase`
    return do_uppercase(output, dataset)