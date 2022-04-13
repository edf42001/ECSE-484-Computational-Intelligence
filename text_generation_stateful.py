from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

from custom_text_gen_callback import CustomTextGenCallback
from custom_reset_state_callback import CustomResetStateCallback
from prepare_training_data_from_file import stateful_training_data_from_file

# Tutorial here:
# https://keras.io/examples/generative/lstm_character_level_text_generation/

base_path = "./"
data_path = base_path + "input/courses/csds.txt"

batch_size = 128
sequence_len = 40

epochs = 60

gen_text_freq = 20
gen_text_length = 150

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len, validation_split=0.05)
X_train, Y_train, X_validate, Y_validate, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)

# Keras model checkpoint saves every N batches, so we need batches/epoch
epoch_save_freq = 3
batches_per_epoch = X_train.shape[0] // batch_size

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

# Freq is measured in batches
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_best_only=False,
    verbose=True,
    # save_freq=epoch_save_freq*batches_per_epoch+1)
    save_freq="epoch")

# Create a custom callback to generate sample text every couple of epochs
custom_text_gen_cb = CustomTextGenCallback(gen_text_freq, gen_text_length, text, char_indices, indices_char)

# Reset the model every epoch, so the statefulness isn't messed up
custom_reset_state_cb = CustomResetStateCallback()

# TODO: use adam?
optimizer = keras.optimizers.RMSprop(learning_rate=0.01, clipnorm=5)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Print a summary of our model
model.summary()

# If the custom_text_gen_cb comes after the model_checkpoint_cb, then the history dictionary becomes empty
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                    callbacks=[model_checkpoint_cb, custom_reset_state_cb, custom_text_gen_cb],
                    shuffle=False, validation_data=(X_validate, Y_validate))

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
