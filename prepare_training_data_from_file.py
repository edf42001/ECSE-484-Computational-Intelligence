import io
import numpy as np


def char_array_to_one_hot(X, char_indices):
    samples, sequence_len = X.shape

    num_chars = len(char_indices)
    # TODO: Investigate: is dtype=bool why my google notebook is running twice as fast? Or did they allocate me a better gpu?
    one_hot = np.zeros((samples, sequence_len, num_chars), dtype=bool)

    for i in range(samples):
        for t in range(sequence_len):
            one_hot[i, t, char_indices[X[i, t]]] = 1

    return one_hot


def get_val_split(X, sequence_length, val_split):
    batch_size, total_length = X.shape

    skim_amount = total_length * val_split
    skim_amount = int(sequence_length * np.ceil(skim_amount / sequence_length))

    # Returns: x_train, x_val. Skim a bit of each sequence off the end to use as validation
    return X[:, :-skim_amount], X[:, -skim_amount:]


def interleave(X, batch_size, sequence_len):
    num_batches = X.shape[1] // sequence_len

    # Interleave the values into the array such that the batches are lined up
    X2 = np.zeros((num_batches, batch_size, sequence_len), dtype='str')
    for i, x in enumerate(X):
        something = np.expand_dims(x.reshape(-1, sequence_len), axis=1)
        X2[:, i::batch_size, :] = something

    return X2.reshape((-1, sequence_len))


def divide_text_into_sequences(text: str, num_batches: int, batch_size: int,
                               sequence_len: int, val_split):
    # Split the input array into batch_size long sequences
    # These are the sequences that a copy of each stateful LSTM will train on in batches in parallel
    X = np.array([c for c in text])
    X = X.reshape((batch_size, -1))

    # Now that we have these sequences, reserve some off the end of each for validation
    # It's not random, but as long as there isn't an even divisible pattern in the text,
    # (vanishingly likelky), then these sequences off the end represent a pretty random sample
    X_train, X_val = get_val_split(X, sequence_len, val_split)

    # Now, interleave these sequences such that when batches by keras, each index of each
    # batch is the continuing sequence of the same index of the previous batch,
    # # meaning wherever we split a sequence it is interleaved with batch_size
    # other sequences. (for a stateful LSTM to work)
    X2_train = interleave(X_train, batch_size, sequence_len)
    X2_val = interleave(X_val, batch_size, sequence_len)

    return X2_train, X2_val


def stateful_training_data_from_file(filename: str, batch_size: int = 128, sequence_len: int = 50,
                                     val_split: float = 0.05, lower: bool = True):
    print("Reading data from " + filename)
    with io.open(filename, encoding="utf-8") as f:
        text = f.read()

    # Optionally can convert to lowercase
    if lower:
        text = text.lower()

    text_len = len(text)
    print("Text length:", text_len)

    num_batches = text_len // (batch_size * sequence_len)

    # Chop off the end so we can evenly divide into batch_size * sequence_len
    # Round down to nearest multiple of batch_size * sequence_len, +1 so the outputs can be the same length
    chop_len = batch_size * sequence_len * num_batches + 1
    text = text[:chop_len]
    text_len = len(text)
    print("Text length after chopping:", text_len)

    # Extract chars and create lookup tables for indices
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    num_chars = len(chars)
    print("Number of different chars:", num_chars)

    X_train, X_val = divide_text_into_sequences(text[:-1], num_batches, batch_size, sequence_len, val_split)
    Y_train, Y_val = divide_text_into_sequences(text[1:], num_batches, batch_size, sequence_len, val_split)

    X_train = char_array_to_one_hot(X_train, char_indices)
    X_val = char_array_to_one_hot(X_val, char_indices)
    Y_train = char_array_to_one_hot(Y_train, char_indices)
    Y_val = char_array_to_one_hot(Y_val, char_indices)

    print("X/Y train shape:", X_train.shape)
    print("X/Y val shape: ", X_val.shape)

    return X_train, Y_train, X_val, Y_val, char_indices, indices_char, text, chars


if __name__ == "__main__":
    filename = "input/alphabet.txt"

    stateful_training_data_from_file(filename, batch_size=5, sequence_len=5, val_split=0.05)

