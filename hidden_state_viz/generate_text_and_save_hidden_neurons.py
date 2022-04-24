from tensorflow import keras
from keras.layers import LSTM, Input, Dense
from keras.models import Model

import numpy as np
import random
import time

from callbacks.custom_text_gen_callback import sample
from prepare_training_data_from_file import stateful_training_data_from_file

base_path = "../"
# data_path = base_path + "input/a_few_courses.txt"
data_path = base_path + "input/all_courses.txt"
# data_path = base_path + "input/good_python.txt"


batch_size = 128
sequence_len = 40

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len, val_split=0.05, lower=True)
_, _, _, _, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)
print(chars)
print(num_chars)

# model_path = base_path + "model_checkpoints/all_courses/150-0.80-0.79.hdf5"
model_path = base_path + "model_checkpoints/good_python/80-0.96-0.96.hdf5"
# model_path = base_path + "model_checkpoints/all_courses_upper/150-0.81-0.80.hdf5"


model = keras.models.load_model(model_path)
model.summary()
print()

model_lstm1 = model.layers[0]
model_lstm2 = model.layers[2]  # There is a dropout layer at index 1
model_lstm3 = model.layers[4]  # There is a dropout layer at index 1

units = model_lstm1.units

# We make a new model with a batch and sequence_len size of 1, because we are feeding in chars
# 1 at a time. This is also faster. If you have a GPU, you could probably generate text in parrallel,
# Although it wouldn't be one long sequence.
inputs = Input(batch_input_shape=(1, 1, model_lstm1.input_shape[2]))
lstm1, state_h1, state_c1 = LSTM(units, return_sequences=True, return_state=True, stateful=True)(inputs)
lstm2, state_h2, state_c2 = LSTM(units, return_sequences=True, return_state=True, stateful=True)(lstm1)
lstm3, state_h3, state_c3 = LSTM(units, return_sequences=True, return_state=True, stateful=True)(lstm2)
# out = Dense(num_chars, activation="softmax")(lstm2)
out = Dense(num_chars, activation="softmax")(lstm3)
# model3 = Model(inputs=[inputs], outputs=[out, state_h1, state_h2])
model3 = Model(inputs=[inputs], outputs=[out, state_h1, state_h2, state_h3])


model3.set_weights(model.get_weights())
model3.summary()
print()

diversity = 0.75
length = 1000

save_hidden_states = True

start = time.time()

# The resulting text
generated = ""

# Sample a sentence to warmup the nketwork with
start_index = random.randint(0, len(text) - sequence_len - 1)
sentence = text[start_index:start_index + sequence_len]

# The output of our model is [logits, hidden1, hidden2, ...]
# Only allocate this memory if we are going to save the results
stored_hidden_states = None
if save_hidden_states:
    num_layers = len(model3.output_shape) - 1
    stored_hidden_states = np.empty((length, num_layers * units))

print("Diversity:", diversity)
print("Generating with: " + sentence)

# Feed the network the sampled sentence as a warmup
for i in range(len(sentence) - 1):
    x_pred = np.zeros((1, 1, len(char_indices)))
    x_pred[0, 0, char_indices[sentence[i]]] = 1.0

    model3.predict(x_pred)

# Now, starting with the last char, feed the chars and sample a char as output
next_char = sentence[-1]

for i in range(length):
    # One hot encoded vector for input char
    x_pred = np.zeros((1, 1, len(char_indices)))
    x_pred[0, 0, char_indices[next_char]] = 1.0

    # Get predictions for our base model, and the model with the same weights that returns hidden states
    out = model3.predict(x_pred)

    preds = out[0][0, 0]
    hidden_states = out[1:]

    # Save hidden weights in our matrix
    if save_hidden_states:
        for j, state in enumerate(hidden_states):
            stored_hidden_states[i, j*units:(j+1)*units] = state[0]

    # Sample a random char
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    generated += next_char

    # To print generated text as it is being created
    # print(next_char, end='', flush=True)

model3.reset_states()

end = time.time()

print("Took: {:.3f}s".format(end - start))
print("Result: " + generated)

# Save the text and (optionally) hidden weights
folder = "hidden_state_viz_data"
hidden_data_file = folder + "/hidden_data.csv"
text_file = folder + "/generated_text.txt"

# Save text
with open(text_file, 'w') as f:
    f.write(generated)

# Save weights
if save_hidden_states:
    np.savetxt(hidden_data_file, stored_hidden_states, delimiter=",")
