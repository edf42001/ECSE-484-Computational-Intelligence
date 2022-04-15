from tensorflow import keras
from keras.layers import LSTM, Input, Dense
from keras.models import Model

import numpy as np
import random
import time

from custom_text_gen_callback import sample
from prepare_training_data_from_file import stateful_training_data_from_file

base_path = "../"
# data_path = base_path + "input/courses/csds.txt"
data_path = base_path + "input/4.1_python.txt"
# data_path = base_path + "input/a_few_courses.txt"

batch_size = 128
sequence_len = 40

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len, validation_split=0.05)
_, _, _, _, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)

model_path = base_path + "model_checkpoints/stateful_3_512_100_1e-4_4.1_python/99-1.16-1.05.hdf5"
# model_path = base_path + "model_checkpoints/stateful_128_a_few_courses/150-1.20-1.13.hdf5"
# model_path = base_path + "model_checkpoints/stateful_2_512_100_few_courses_lr_decay/250-1.06-1.06.hdf5"

model = keras.models.load_model(model_path)
model.summary()
print()

model_lstm1 = model.layers[0]
model_lstm2 = model.layers[2]  # There is a dropout layer at index 1
model_lstm3 = model.layers[4]  # There is a dropout layer at index 1

units = model_lstm1.units

inputs = Input(batch_input_shape=(1, 1, model_lstm1.input_shape[2]))
lstm1, state_h1, state_c1 = LSTM(model_lstm1.units, return_sequences=True, return_state=True, stateful=True)(inputs)
lstm2, state_h2, state_c2 = LSTM(model_lstm1.units, return_sequences=True, return_state=True, stateful=True)(lstm1)

# 3rd layer
lstm3, state_h3, state_c3 = LSTM(model_lstm1.units, return_sequences=True, return_state=True, stateful=True)(lstm2)

# model2 = Model(inputs=[inputs], outputs=[lstm1, state_h1, state_c1, lstm2, state_h2, state_c2])
model2 = Model(inputs=[inputs], outputs=[lstm1, state_h1, state_c1, lstm2, state_h2, state_c2, lstm3, state_h3, state_c3])

# For faster computation, don't need to compute batch size of 128
out = Dense(num_chars, activation="softmax")(lstm2)
model3 = Model(inputs=[inputs], outputs=[out])

model2.layers[1].set_weights(model_lstm1.get_weights())
model2.layers[2].set_weights(model_lstm2.get_weights())
model2.layers[3].set_weights(model_lstm3.get_weights())

model2.summary()
print()


# model3.set_weights(model.get_weights())
model3.summary()
print()

# # because including input layer, no dropout. 4 because including dropout, but no input layer
n_hidden_vectors = 9
# n_hidden_vectors = 6


diversity = 0.85  # For good python results
# diversity = 0.8 # For good courses results
length = 100

start = time.time()

# The resulting text
generated = ""

# Sample a sentence to warmup the network with
start_index = random.randint(0, len(text) - sequence_len - 1)
sentence = text[start_index:start_index + sequence_len]

stored_hidden_states = np.empty((length, n_hidden_vectors * units))

print("Diversity:", diversity)
print("Generating with: " + sentence)

# Feed the network the sampled sentence as a warmup
for i in range(len(sentence) - 1):
    x_pred = np.zeros((1, 1, len(char_indices)))
    x_pred[0, 0, char_indices[sentence[i]]] = 1.0

    model.predict(x_pred)
    model2.predict(x_pred)

# Now, starting with the last char, feed the chars and sample a char as output
next_char = sentence[-1]

for i in range(length):
    # One hot encoded vector for input char
    x_pred = np.zeros((1, 1, len(char_indices)))
    x_pred[0, 0, char_indices[next_char]] = 1.0

    # Get predictions for our base model, and the model with the same weights that returns hidden states
    preds = model.predict(x_pred)[0, 0]
    hidden_states = model2.predict(x_pred)

    # Save hidden weights in our matrix
    for j in range(n_hidden_vectors):
        if len(hidden_states[j].shape) == 3:
            # return sequences is true so we need to get only the last value
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j][0, -1, :]
        else:
            stored_hidden_states[i, j*units:(j+1)*units] = hidden_states[j][0]

    # Sample a random char
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    generated += next_char

model.reset_states()
model2.reset_states()

end = time.time()

print("Took: {:.3f}s".format(end - start))
print("Result: " + generated)

# Save the text and hidden weights
folder = "hidden_state_viz_data"
hidden_data_file = folder + "/hidden_data.csv"
text_file = folder + "/generated_text.txt"

np.savetxt(hidden_data_file, stored_hidden_states, delimiter=",")
with open(text_file, 'w') as f:
    f.write(generated)

