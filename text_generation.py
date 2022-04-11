from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

import numpy as np
import random
import io
import os
from datetime import datetime

from custom_text_gen_callback import CustomTextGenCallback

# Tutorial here:
# https://keras.io/examples/generative/lstm_character_level_text_generation/

base_path = "./"
data_path = base_path + "input/csds.txt"

# Read the file, all lowercase text to start with
with io.open(data_path, encoding="utf-8") as f:
    text = f.read().lower()
print("Text length:", len(text))

# Extract the number of different characters in the text
# Create lookup tables for character/index
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
num_chars = len(chars)
print("Number of different chars:", num_chars)

# Split the text into chunks. The chunks overlap. This helps training?
maxlen = 40
step = 3
sentences = []
next_chars = []  # The next character given the previous maxlen context characters
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])

print("Number of sequences:", len(sentences))
# print(sentences)
# print(next_chars)

# One hot encoded input sequences and output character
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)

# Create the one hot encodings
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Model: a single LSTM layer TODO? Add dropout
model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, num_chars)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(num_chars, activation="softmax")
    ]
)

# Create directory for saving if does not exist
if not os.path.isdir(base_path + "model_checkpoints"):
    print("model_checkpoints directory not found, creating")
    os.mkdir(base_path + "model_checkpoints")


# Create a model checkpoint that will save all models to a folder with the current time,
# and a file name with epoch, loss, and validation loss in it
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.mkdir(base_path + "model_checkpoints" + "/" + current_time)
checkpoint_filepath = os.path.join(base_path, "model_checkpoints", current_time, "{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5")

model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_best_only=False,
    verbose=True)

# Create a custom callback to generate sample text every couple of epochs
custom_text_gen_cb = CustomTextGenCallback(2, 10, maxlen, text, chars, char_indices, indices_char)

# TODO: use adam?
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Print a summary of our model
model.summary()

epochs = 1
# batch_size = 128
batch_size = 2048

# If the custom_text_gen_cb comes after the model_checkpoint_cb, then the history dictionary becomes empty
history = model.fit(x, y, batch_size=batch_size, epochs=epochs,
                    callbacks=[custom_text_gen_cb, model_checkpoint_cb],
                    validation_split=0.1)

print(history.history)
