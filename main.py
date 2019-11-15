import tensorflow as tf

import json

import progressbar

from tensorflow.keras import layers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import pickle
import time

BATCH_SIZE = 16
# BUFFER_SIZE = 2**20

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', dest='dev', action='store_true')
parser.add_argument('--config', dest='config_path', type=str, required=True)
args = parser.parse_args()

with open(args.config_path,'r') as ifile:
    CONFIG_JSON = json.load(ifile)['development' if args.dev else 'production']

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # Create some tensors
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

    print(c)
    input("Input:")