from tensorflow import keras
import numpy as np
import random


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_samples(model, length, char_indices, indices_char):
    # Save model history, because calling predict or evaluate clears it
    history = model.history

    # Generate a few examples with different sizes
    for diversity in [0.5, 1.0]:
        print("Diversity:", diversity)

        generated = ""

        # Next char to feed the network
        next_char = "t"
        for i in range(length):
            x_pred = np.zeros((1, 1, len(char_indices)))
            x_pred[0, 0, char_indices[next_char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0, 0]

            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char

        print("Generated:", generated)
        print()

        # Reset states when done, since model is stateful
        model.reset_states()

    # Backup model history
    model.history = history

    return generated


class CustomTextGenCallback(keras.callbacks.Callback):
    def __init__(self, epoch_frequency, length, char_indices, indices_char):
        self.epoch_frequency = epoch_frequency

        # Need all this stuff to do the sampling. Does this double data storage usage, or is it pass by reference?
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.length = length

        self.pred_model = None

    def on_epoch_end(self, epoch, logs=None):
        # Epochs start counting at 1, so subtract 1
        if epoch % self.epoch_frequency == 0:
            self.create_pred_model()
            generate_samples(self.pred_model, self.length, self.char_indices, self.indices_char)

    def on_train_end(self, logs=None):
        # In case the frequencies don't line up, also want to call this when we finish training for final results
        self.create_pred_model()
        generate_samples(self.pred_model, self.length, self.char_indices, self.indices_char)

    def create_pred_model(self):
        # Need to make a copy of the model that has batch size and sequence length of 0 so we can
        # have it predict characters one at a time, otherwise tensorflow doesn't like it.
        # Does this double RAM usage?
        units = self.model.layers[0].units

        # This needs to work automatically for both 2 & 3 layer LSTMs
        self.pred_model = keras.Sequential(
            [
                keras.Input(batch_input_shape=(1, 1, len(self.char_indices))),
                keras.layers.LSTM(units, return_sequences=True, stateful=True),
                keras.layers.LSTM(units, return_sequences=True, stateful=True),
                keras.layers.LSTM(units, return_sequences=True, stateful=True),
                keras.layers.Dense(len(self.char_indices), activation="softmax")
            ]
        )

        self.pred_model.set_weights(self.model.get_weights())
