from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import LSTM, Input, Dense
from keras.models import Model

import numpy as np
import random
import io

from custom_text_gen_callback import sample
from prepare_training_data_from_file import stateful_training_data_from_file

base_path = "../"
data_path = base_path + "input/courses/csds.txt"

batch_size = 128
sequence_len = 40

epochs = 60

gen_text_freq = 20
gen_text_length = 150

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len, validation_split=0.05)
X_train, Y_train, X_validate, Y_validate, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)


model_path = base_path + "model_checkpoints/20220413_153403/40-1.06-1.31.hdf5"
model = keras.models.load_model(model_path)
model.summary()
print()

model_lstm1 = model.layers[0]
model_lstm2 = model.layers[2]  # There is a dropout layer at index 1
units = model_lstm1.units

inputs = Input(batch_input_shape=(1, model_lstm1.input_shape[1], model_lstm1.input_shape[2]))
lstm1, state_h1, state_c1 = LSTM(model_lstm1.units, return_sequences=True, return_state=True, stateful=True)(inputs)
lstm2, state_h2, state_c2 = LSTM(model_lstm1.units, return_sequences=True, return_state=True, stateful=True)(lstm1)

model2 = Model(inputs=[inputs], outputs=[lstm1, state_h1, state_c1, lstm2, state_h2, state_c2])

# For faster computation, don't need to compute batch size of 128
out = Dense(num_chars, activation="softmax")(lstm2)
model3 = Model(inputs=[inputs], outputs=[out])

model2.layers[1].set_weights(model_lstm1.get_weights())
model2.layers[2].set_weights(model_lstm2.get_weights())
model2.summary()
print()


model3.set_weights(model.get_weights())
model3.summary()
print()

# # because including input layer, no dropout. 4 becuase including dropout, but no input layer
n_hidden_vectors = 6

diversity = 0.7
length = 400
print("Diversity:", diversity)

generated = ""
start_index = random.randint(0, len(text) - sequence_len - 1)
sentence = text[start_index:start_index + sequence_len]
print("Generating with: " + sentence)

stored_hidden_states = np.empty((length, n_hidden_vectors * units))

next_char = "t"
for i in range(length):
    x_pred = np.zeros((1, 1, len(char_indices)))
    x_pred[0, 0, char_indices[next_char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0, 0]

    hidden_states = model2.predict(x_pred)

    for j in range(n_hidden_vectors):
        if len(hidden_states[j].shape) == 3:
            # return sequences is true so we need to get only the last value
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j][0, -1, :]
        else:
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j][0]

    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    generated += next_char

print("Result: " + generated)

folder = "hidden_state_viz_data"
hidden_data_file = folder + "/hidden_data.csv"
text_file = folder + "/generated_text.txt"

np.savetxt(hidden_data_file, stored_hidden_states, delimiter=",")
with open(text_file, 'w') as f:
    f.write(generated)
