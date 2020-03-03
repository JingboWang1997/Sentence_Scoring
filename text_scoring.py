# https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470
# https://www.tensorflow.org/tutorials/text/text_classification_rnn

from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
import json

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

# train the model
print(np.array(padded_sequences))
print(np.array(labels))
history = model.fit(
	x=np.array(padded_sequences),
	y=np.array(labels),
	epochs=10)


