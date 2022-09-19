print('Importing modules...')
import numpy as np
import pandas as pd
import string
import os
import tensorflow as tf
import keras
from keras.utils import pad_sequences

def strip_punctuation(s: str):
    return s.translate(s.maketrans({i:'' for i in string.punctuation})).lower()

os.system('cls||clear')

classes = ('List', 'Power number', 'Get their attention', '2 nouns 2 commas', 'occasion position')

data = pd.read_csv('./data.csv', delimiter=';')
labels = data.pop('labels').to_numpy(dtype=np.int8).flatten()
data = data.to_numpy().flatten()


PAD = 0

word2idx = {i+1:strip_punctuation(v) for i,v in enumerate(sorted(set(' '.join(data).split())))}
char2idx = {v:i for i,v in word2idx.items()}

VOCAB_SIZE = len(char2idx)

def encode_str(string: str):
    '''Encodes the string passed.'''
    return np.array(pad_sequences([[char2idx[i.lower()] if i.lower() in char2idx else PAD for i in string.replace(',','').split()]], 250, 'int8'), dtype=np.int8)

def encode(arr: np.ndarray):
    '''Ecnodes the arr passed.'''
    return np.array(pad_sequences([encode_str(i).flatten() for i in arr.flatten()], 250, 'int8'), dtype=np.int8)

data = encode(data)

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(250,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, 256),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax'),
])
model.compile(optimizer='adamax', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), jit_compile=True)
model.fit(data, labels, epochs=10)
model.save('./weights.h5')

def predict(string: str):
    prediction = model.predict(encode_str(strip_punctuation(string))).flatten()
    return classes[sorted(enumerate(prediction), key=lambda x: x[1], reverse=True)[0][0]], prediction

print(predict('john doe, a cookie baker, loves cookies'))
