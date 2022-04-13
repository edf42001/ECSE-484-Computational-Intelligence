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


def generate_samples(model, length, text, char_indices, indices_char):
    # Save model history, because calling predict or evaluate clears it
    history = model.history

    start_index = random.randint(0, len(text)-1)
    for diversity in [0.5, 1.0]:
        print("Diversity:", diversity)

        generated = ""
        # sentence = text[start_index : start_index + maxlen]
        # print('Generating with seed: "' + sentence + '"')

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
    def __init__(self, epoch_frequency, length, text, char_indices, indices_char):
        self.epoch_frequency = epoch_frequency

        # Need all this stuff to do the sampling. Does this double data storage usage, or is it pass by reference?
        self.text = text
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.length = length

    def on_epoch_end(self, epoch, logs=None):
        # Epochs start counting at 1, so subtract 1
        if epoch % self.epoch_frequency == 0:
            generate_samples(self.model, self.length, self.text, self.char_indices, self.indices_char)

    def on_train_end(self, logs=None):
        # In case the frequencies don't line up, also want to call this when we finish training for final results
        generate_samples(self.model, self.length, self.text, self.char_indices, self.indices_char)
