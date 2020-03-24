# https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
# https://www.tensorflow.org/tutorials/text/text_classification_rnn

from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import json
import os

'''DATA'''
print('loading data...')
raw_data = None
with open("../../Data/reddit_jokes.json", "r") as read_file:
    raw_data = json.load(read_file)
'''each in raw_data
{'body': 'Now I have to say "Leroy can you please paint the fence?"', 'id': '5tz52q', 'score': 1, 'title': 'I hate how you cant even say black paint anymore'} 
'''

features = []
labels = []
for each in raw_data:
    features.append(each['title'] + ' ' + each['body'])
    labels.append(each['score'])
labels = np.tanh(labels)

print('processing data...')
# Create Tokenizer Object
tokenizer = Tokenizer(
	num_words=10000,
	filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
	lower=True,
	split=' ')

# Train the tokenizer to the texts
tokenizer.fit_on_texts(features)

# Convert list of strings into list of lists of integers
sequences = tokenizer.texts_to_sequences(features)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
# max length = 7405
# split training and testing
train_len = int(len(padded_sequences) * 0.8)
train_data = padded_sequences[0:train_len]
train_labels = labels[0:train_len]
test_data = padded_sequences[train_len:len(padded_sequences)]
test_labels = labels[train_len:len(padded_sequences)]

'''MODEL'''
print('constructing model...')
# building RNN
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.num_words, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
	loss=tf.keras.losses.MeanSquaredError(),
	optimizer=tf.keras.optimizers.Adam(1e-4),
	metrics=['mae'])
# checkpoint setup
# Create a callback that saves the model's weights
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# train the model
try:
    print('using checkpoint saved weights...')
    model.load_weights(checkpoint_path)
except:
    print('no saved weights found, starting new...')
    pass
history = model.fit(
        train_data,
	train_labels,
	epochs=10,
        validation_data=(test_data,test_labels),
        callbacks=[cp_callback])


