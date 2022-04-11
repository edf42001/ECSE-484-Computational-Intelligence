from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Input
from keras.models import Model

import numpy as np
import random
import io
import os

from custom_text_gen_callback import sample, generate_samples

base_path = "../"
data_path = base_path + "input/csds.txt"

# Process the data
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

maxlen = 40

model = keras.models.load_model(base_path + "model_checkpoints/20220410_195605/01-3.25-3.12.hdf5")
model.summary()

model_lstm1 = model.layers[0]

print(model_lstm1)
print(model_lstm1.input_shape)

inputs = Input(shape=model_lstm1.input_shape[1:])
lstm, state_h, state_c = LSTM(model_lstm1.units, return_state=True)(inputs)
model2 = Model(inputs=[inputs], outputs=[lstm, state_h, state_c])
model2.layers[1].set_weights(model_lstm1.get_weights())

diversity = 1.0
length = 100
print("...Diversity:", diversity)

generated = ""
start_index = random.randint(0, len(text) - maxlen - 1)
sentence = text[start_index:start_index + maxlen]
print("Generating with: " + sentence)

stored_hidden_states = np.empty((length, 3 * 64))

for i in range(length):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]

    hidden_states = model2.predict(x_pred)
    for j in range(3):
        stored_hidden_states[i, j*64:(j+1)*64] = hidden_states[j]

    layer1 = model.layers[0]
    layer2 = model.layers[1]

    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    sentence = sentence[1:] + next_char
    generated += next_char


print("Result: " + generated)

folder = "hidden_state_viz_data"
hidden_data_file = folder + "/hidden_data.csv"
text_file = folder + "/generated_text.txt"

np.savetxt(hidden_data_file, stored_hidden_states, delimiter=",")
with open(text_file, 'w') as f:
    f.write(generated)
