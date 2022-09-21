print('Importing modules...')
import numpy as np
import pandas as pd
import string
import os
import tensorflow as tf
import keras
from keras.utils import pad_sequences

def strip_punctuation(s: str):
    return s.lower()
    # return s.translate(s.maketrans({i:'' for i in string.punctuation})).lower()

os.system('cls||clear')

classes = ('List', 'Power number', 'Get their attention', '2 nouns 2 commas', 'occasion position')

data = pd.read_csv('./data.csv', delimiter=';')
labels = data.pop('labels').to_numpy(dtype=np.int8).flatten()
data = data.to_numpy().flatten()


PAD = 0

word2idx = {i+1:v.lower() for i,v in enumerate(sorted(set('\n'.join(data))))}
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
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation='softmax'),
])
model.compile(optimizer='nadam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), jit_compile=True)
model.fit(data, labels, epochs=17)
model.save('./weights.h5')

def predict(string: str):
    prediction = model.predict(encode_str(string.lower())).flatten()
    return tuple(classes[i] for i in reversed(np.argsort(prediction))), prediction

print(predict('Popeyes is the best restaurant because, its tasty, fast, and clean.'))
print(predict('john, a boy, loves fruits!'))
print(predict('when I get home, I clean my room.'))
print(predict('the goverment is good for 3 reasons.'))
print(predict('KFC is the worst store!'))
print(predict('dates are a healthy snack and fruit.'))