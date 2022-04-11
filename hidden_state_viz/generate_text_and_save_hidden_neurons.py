from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Input
from keras.models import Model

import numpy as np
import random
import io

from custom_text_gen_callback import sample, generate_samples

base_path = "../"
data_path = base_path + "input/all.txt"

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

model_path = base_path + "model_checkpoints/all_2lstm_128_dropout_adam/30-1.30-1.21.hdf5"
model = keras.models.load_model(model_path)
model.summary()

model_lstm1 = model.layers[0]
model_lstm2 = model.layers[2]  # There is a dropout layer at index 1
units = model_lstm1.units

inputs = Input(shape=model_lstm1.input_shape[1:])
lstm1, state_h1, state_c1 = LSTM(model_lstm1.units, return_sequences=True, return_state=True)(inputs)
lstm2, state_h2, state_c2 = LSTM(model_lstm1.units, return_state=True)(lstm1)
model2 = Model(inputs=[inputs], outputs=[lstm1, state_h1, state_c1, lstm2, state_h2, state_c2])
model2.layers[1].set_weights(model_lstm1.get_weights())
model2.layers[2].set_weights(model_lstm2.get_weights())
model2.summary()
n_hidden_vectors = 6

diversity = 0.55
length = 550
print("Diversity:", diversity)

generated = ""
start_index = random.randint(0, len(text) - maxlen - 1)
sentence = text[start_index:start_index + maxlen]
print("Generating with: " + sentence)

stored_hidden_states = np.empty((length, n_hidden_vectors * units))

for i in range(length):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]

    hidden_states = model2.predict(x_pred)

    for j in range(n_hidden_vectors):
        if j == 0:
            # return sequences is true so we need to get only the last value
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j][0, -1, :]
        else:
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j]

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
