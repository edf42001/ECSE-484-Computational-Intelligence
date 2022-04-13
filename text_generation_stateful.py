from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

import numpy as np
import random
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt

from custom_text_gen_callback import sample
from custom_text_gen_callback import CustomTextGenCallback
from custom_reset_state_callback import CustomResetStateCallback
from prepare_training_data_from_file import stateful_training_data_from_file

# Tutorial here:
# https://keras.io/examples/generative/lstm_character_level_text_generation/

base_path = "./"
data_path = base_path + "input/courses/csds.txt"

batch_size = 128
sequence_len = 40

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len, validation_split=0.05)
X_train, Y_train, X_validate, Y_validate, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)

# Model: a single LSTM layer TODO? Add dropout
model = keras.Sequential(
    [
        keras.Input(batch_input_shape=(batch_size, sequence_len, num_chars)),  # shape=(maxlen, num_chars),
        layers.LSTM(64, return_sequences=True, stateful=True),
        layers.Dropout(0.0),
        layers.LSTM(64, return_sequences=True, stateful=True),
        layers.Dropout(0.0),
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
custom_text_gen_cb = CustomTextGenCallback(2, 50, sequence_len, text, chars, char_indices, indices_char)

# Reset the model every epoch, so the statefulness isn't messed up
custom_reset_state_cb = CustomResetStateCallback()

# TODO: use adam?
optimizer = keras.optimizers.RMSprop(learning_rate=0.01, clipnorm=5)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Print a summary of our model
model.summary()

epochs = 30

# If the custom_text_gen_cb comes after the model_checkpoint_cb, then the history dictionary becomes empty
for i in range(1):
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=[model_checkpoint_cb, custom_reset_state_cb], shuffle=False,
                        validation_data=(X_validate, Y_validate))
    print(history.history)

    if i % 1 == 0:
        generated = ""
        start_index = random.randint(0, len(text) - sequence_len - 1)
        sentence = text[start_index:start_index + sequence_len]
        print("Generating with: " + sentence)

        diversity = 0.5
        length = 100
        next_char = "t"
        for i in range(length):
            x_pred = np.zeros((1, 1, num_chars))
            x_pred[0, 0, char_indices[next_char]] = 1.0
            # print(x_pred.shape)

            # for t, char in enumerate(sentence):

            preds = model.predict(x_pred, verbose=0)
            preds = preds[0, 0]
            # print(x_pred.shape)
            # print(preds, preds.shape)
            # print(preds[0, 0])

            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print(generated)

        model.reset_states()


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
