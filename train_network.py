from tensorflow import keras
from tensorflow.keras import layers

import os
from datetime import datetime
import matplotlib.pyplot as plt

from callbacks.custom_text_gen_callback import CustomTextGenCallback
from callbacks.custom_reset_state_callback import CustomResetStateCallback
from prepare_training_data_from_file import stateful_training_data_from_file

# Tutorial here:
# https://keras.io/examples/generative/lstm_character_level_text_generation/

base_path = "./"

# data_path = base_path + "input/all_courses.txt"
data_path = base_path + "input/good_python.txt"

# Name of folder to save models in
model_checkpoint_folder = "good_python"

batch_size = 128
sequence_len = 50

epochs = 60
epoch_save_freq = 5

gen_text_freq = 20
gen_text_length = 100

ret = stateful_training_data_from_file(data_path, batch_size, sequence_len,
                                       val_split=0.05, lower=True)
X_train, Y_train, X_validate, Y_validate, char_indices, indices_char, text, chars = ret
num_chars = len(char_indices)
print("Chars:", chars)

# Optionally, can load from folder
load = False
load_path = base_path + "model_checkpoints/" + "good_python/" + "80-0.96-0.96.hdf5"

if load:
    print("Loading model from " + load_path)
    model = keras.models.load_model(load_path)
else:
    # Model: a 3 layer LSTM with dropout
    model = keras.Sequential(
        [
            keras.Input(batch_input_shape=(batch_size, sequence_len, num_chars)),
            layers.LSTM(512, return_sequences=True, stateful=True),
            layers.Dropout(0.5),
            layers.LSTM(512, return_sequences=True, stateful=True),
            layers.Dropout(0.5),
            layers.LSTM(512, return_sequences=True, stateful=True),
            layers.Dropout(0.5),
            layers.Dense(num_chars, activation="softmax")
        ]
    )

# Create directory for saving if does not exist
if not os.path.isdir(base_path + "model_checkpoints"):
    print("model_checkpoints directory not found, creating")
    os.mkdir(base_path + "model_checkpoints")

# Create a model checkpoint that will save all models to a folder with the current time,
# and a file name with epoch, loss, and validation loss in it
if not os.path.isdir(base_path + "model_checkpoints" + "/" + model_checkpoint_folder):
    os.mkdir(base_path + "model_checkpoints" + "/" + model_checkpoint_folder)

checkpoint_filepath = os.path.join(base_path, "model_checkpoints", model_checkpoint_folder, "{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5")

# Freq is measured in batches
model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_best_only=False,
    verbose=True,
    period=epoch_save_freq  # period is deprecated but this is the only way to get "val_loss" into the format string
)

# Create a custom callback to generate sample text every couple of epochs
custom_text_gen_cb = CustomTextGenCallback(gen_text_freq, gen_text_length, char_indices, indices_char)

# Reset the model every epoch, so the statefulness isn't messed up
custom_reset_state_cb = CustomResetStateCallback()


# Custom learning rate schedule (step down every couple of epochs)
# Doesn't seem to help much
def schedule(epoch, lr):
    decay = 0.2
    if epoch == 100:
        return lr * decay
    else:
        return lr


learn_rate_schedule_cb = keras.callbacks.LearningRateScheduler(schedule, verbose=True)


optimizer = keras.optimizers.RMSprop(learning_rate=0.00015, clipnorm=5)
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
