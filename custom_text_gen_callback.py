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


def generate_samples(model, maxlen, text, chars, char_indices, indices_char):
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        print('...Generating with seed: "' + sentence + '"')

        for i in range(100):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print()


class CustomTextGenCallback(keras.callbacks.Callback):
    def __init__(self, epoch_frequency, maxlen, text, chars, char_indices, indices_char):
        self.epoch_frequency = epoch_frequency

        # Need all this stuff to do the sampling. Does this double data storage usage, or is it pass by reference?
        self.maxlen = maxlen
        self.text = text
        self.chars = chars
        self.char_indices = char_indices
        self.indices_char = indices_char

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_frequency == 0:
            generate_samples(self.model, self.maxlen, self.text, self.chars, self.char_indices, self.indices_char)

    def on_train_end(self, logs=None):
        # In case the frequencies don't line up, also want to call this when we finish training for final results
        generate_samples(self.model, self.maxlen, self.text, self.chars, self.char_indices, self.indices_char)
