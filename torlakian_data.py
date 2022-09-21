import os
import sys
from typing import Dict, List, TextIO, Union
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

# Based on: https://github.com/ufal/npfl114/blob/master/labs/03/uppercase_data.py

class TorlakianData:
    LABELS: int = 2

    URL = 'https://www.clarin.si/repository/xmlui/handle/11356/1281/allzip'


    # Class for working with the data in a customized way.
    # - window: int = number of characters left and right from the current position
    # 
    # - alphabet: int  = maximum number of most frequent characters that are going to stay in the data,
    #                    while the rest are replaced by <unk>
    #
    # - clean: bool = states if the data should be cleaned of annotator's marks
    # 
    # - seed: int  = fixing random seed
    class Dataset:
        def __init__(self, data: str, window: int, alphabet: Union[int, List[str]], clean: bool = False, seed: int = 42) -> None:
            if clean:
                data = self._clean_data(data)

            self._window = window
            self._text = data
            self._size = len(self._text)

            # Create alphabet_map
            alphabet_map = {"<pad>": 0, "<unk>": 1}
            if not isinstance(alphabet, int):
                for index, letter in enumerate(alphabet):
                    alphabet_map[letter] = index
            else:
                # Find most frequent characters
                freqs = {}
                for char in self._text.lower():
                    freqs[char] = freqs.get(char, 0) + 1

                most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
                for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                    alphabet_map[char] = i
                    if alphabet and len(alphabet_map) >= alphabet:
                        break

            # Remap lowercased input characters using the alphabet_map
            lcletters = np.zeros(self._size + 2 * window, np.int16)
            for i in range(self._size):
                char = self._text[i].lower()
                if char not in alphabet_map:
                    char = "<unk>"
                lcletters[i + window] = alphabet_map[char]

            # Generate input batches
            windows = np.zeros([self._size, 2 * window + 1], np.int16)
            labels = np.zeros(self._size, np.uint8)
            for i in range(self._size):
                windows[i] = lcletters[i:i + 2 * window + 1]
                labels[i] = self._text[i].isupper()
            self._data = {"windows": windows, "labels": labels}

            # Compute alphabet
            self._alphabet = [None] * len(alphabet_map)
            for key, value in alphabet_map.items():
                self._alphabet[value] = key

        # Cleans the data of annotator notes in the speaker utterances
        def _clean_data(self, data: str) -> str:
            annotations = [
                '((laugh))', 
                '((XXX))',
                '((?))',
                '((NT))',
                '((noise))',
                '((sigh))',
                '((cough))',
                '((cry))',
                '((clears_throat))',
                '((sniffs))',
                '((barking))',
                '((laughs))',
                '((sneeze))',
                '((singing))',
                '((phone_rings))',
                '((birds_chirping))',
                '((knocking))',
                '((inhales))'
            ]

            out_lines = []
            for line in data.split('\n'):
                new_line = line

                # Removing all occurrences of an annotation from the current line
                for annotation in annotations:
                    while annotation in new_line:
                        new_line = new_line.replace(annotation, '')
                        new_line = new_line.strip()
                
                # Using only non-empty lines
                if new_line is not None and len(new_line) != 0:
                    out_lines.append(new_line)

            return '\n'.join(out_lines)

        @property
        def alphabet(self) -> List[str]:
            """List of most frequent characters in the training data"""
            return self._alphabet

        @property
        def text(self) -> str:
            """Original text"""
            return self._text

        @property
        def data(self) -> Dict[str, np.ndarray]:
            """Data in the format of a dictionary with two values:
            - windows: each element is a np array of size 2*windows_size + 1 and represnts
                       characters around the current character
            - labels: boolean value that defines if a character at that position is uppercased or not"""
            return self._data

        @property
        def size(self) -> int:
            """Size of the dataset"""
            return self._size

        @property
        def dataset(self) -> tf.data.Dataset:
            """TF Dataset object"""
            return tf.data.Dataset.from_tensor_slices(self._data)

    def _download(self) -> str:
        """Downloads the corpus"""
        
        directory = "data"
        path = os.path.join(directory, "allzip.zip")
        if not os.path.exists(path):
            os.makedirs(directory)
            print("Downloading dataset {}...".format(path), file=sys.stderr)
            urllib.request.urlretrieve(self.URL, filename=path)
            print("Download finished.")

        return path

    def _unzip(self, path: str) -> None:
        """Unzipping of the corpus .zip archive"""

        with zipfile.ZipFile(path, 'r') as zip_ref:
            directory = os.path.dirname(path)
            zip_ref.extractall(directory)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".zip") and filename != os.path.basename(path):
                full_path = os.path.join(directory, filename)
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(directory)

                os.remove(full_path)
        
    def _join_data(self) -> None:
        """Joins all files in the corpus into one big dataset, 
        then performs 70:20:10 train-val-test split and 
        saves these into separate files"""

        directory = 'data/TOR_C_TXT_transcripts'
        all_data = []

        for file in os.listdir(directory):
            if not file.startswith("TIM_"):
                continue

            filepath = os.path.join(directory, os.fsdecode(file))
            if not filepath.endswith(".txt"):
                continue

            with open(filepath, 'r') as f:
                lines = f.readlines()
                lines = lines[1:]
                lines = [' '.join(l.split()[1:]) for l in lines]
                all_data.extend(lines)

        num_lines = len(all_data)
        train = all_data[:int(num_lines * 0.7)]
        val = all_data[int(num_lines * 0.7): int(num_lines * 0.9)]
        test = all_data[int(num_lines * 0.9):]

        with open('data/train.txt', 'w') as f:
            f.write('\n'.join(train))
        with open('data/val.txt', 'w') as f:
            f.write('\n'.join(val))
        with open('data/test.txt', 'w') as f:
            f.write('\n'.join(test))

    def __init__(self, window: int, alphabet_size: int = 0, clean: bool = False):
        """Constructor for TorlakianData class"""
        path = self._download()
        self._unzip(path)
        self._join_data()

        for dataset in ["train", "val", "test"]:
            with open(f"data/{dataset}.txt", "r") as dataset_file:
                data = dataset_file.read()
            setattr(self, dataset, self.Dataset(
                data,
                window,
                alphabet=alphabet_size if dataset == "train" else self.train.alphabet,
                clean=clean
            ))

    train: Dataset
    val: Dataset
    test: Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(gold_dataset: Dataset, predictions: str) -> float:
        gold = gold_dataset.text

        if len(predictions) < len(gold):
            raise RuntimeError(f"The predictions are shorter than gold data: {len(predictions)} vs {len(gold)}.")

        correct = 0
        for i in range(len(gold)):
            if predictions[i].lower() != gold[i].lower():
                raise RuntimeError(f"The predictions and gold data differ on position {i}: {repr(predictions[i:i+20].lower())} vs {repr(gold[i:i + 20].lower())}.")

            correct += gold[i] == predictions[i]
        return 100.0 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: Dataset, predictions_file: TextIO) -> float:
        predictions = predictions_file.read()
        return TorlakianData.evaluate(gold_dataset, predictions)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--dataset", default="val", type=str, help="Gold dataset to evaluate")
    parser.add_argument("--clean", default=False, help="Clean data", action='store_true')
    args = parser.parse_args()
    
    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = TorlakianData.evaluate_file(getattr(TorlakianData(0, clean=args.clean), args.dataset), predictions_file)
        print("Accent prediction accuracy: {:.2f}%".format(accuracy))
