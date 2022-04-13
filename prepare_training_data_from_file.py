import io
import numpy as np


def divide_text_into_sequences(text: str, num_batches: int, batch_size: int,
                               sequence_len: int, char_indices) -> np.ndarray:
    # Interleave the values into the array such that the batches are lined up
    X = np.array([c for c in text])
    X = X.reshape((batch_size, -1))
    X2 = np.zeros((num_batches, batch_size, sequence_len), dtype='str')

    for i, x in enumerate(X):
        something = np.expand_dims(x.reshape(-1, sequence_len), axis=1)
        X2[:, i::batch_size, :] = something

    # print(X)
    # print(X2)
    # print(X2.shape)

    X2 = X2.reshape((-1, sequence_len))
    # print(X2)
    # print(X2.shape)

    num_chars = len(char_indices)
    X3 = np.zeros((num_batches * batch_size, sequence_len, num_chars))
    for i in range(num_batches * batch_size):
        for t in range(sequence_len):
            X3[i, t, char_indices[X2[i, t]]] = 1

    # print(X3)
    # print(X3.shape)

    return X3


def stateful_training_data_from_file(filename: str, batch_size: int = 128, sequence_len: int = 50, validation_split: float = 0.05):
    print("Reading data from " + filename)
    with io.open(filename, encoding="utf-8") as f:
        text = f.read().lower()

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

    # Split the text into chunks of the input and output sequence. (-1, because that extra 1 is for the output sequence)
    # in_sentences = []
    # out_sentences = []
    # step = sequence_len
    # for i in range(0, len(text) - 1, step):
    #     in_sentences.append(text[i : i + sequence_len])
    #     out_sentences.append(text[i + 1 : i + 1 + sequence_len])

    X = divide_text_into_sequences(text[:-1], num_batches, batch_size, sequence_len, char_indices)
    Y = divide_text_into_sequences(text[1:], num_batches, batch_size, sequence_len, char_indices)

    num_sequences = X.shape[0]
    print("Data shape:", X.shape)

    # print("Number of sequences:", len(in_sentences))
    # print("In:", in_sentences)
    # print("Out:", out_sentences)

    # Pick some indices that will be for validation
    n_validation = num_sequences * validation_split
    n_validation = int(batch_size * round(n_validation / batch_size))  # Round to nearest batch size
    validation_indices = np.random.choice(num_sequences, size=n_validation, replace=False)

    # Extract these sequences for validation
    X_validate = X[validation_indices]
    Y_validate = Y[validation_indices]

    # THe other portion is for training
    X_train = np.delete(X, validation_indices, axis=0)
    Y_train = np.delete(Y, validation_indices, axis=0)

    print(X_train.shape)
    print(Y_train.shape)
    print(X_validate.shape)
    print(Y_validate.shape)

    return X_train, Y_train, X_validate, Y_validate, char_indices, indices_char, text, chars


if __name__ == "__main__":
    filename = "input/alphabet.txt"
    stateful_training_data_from_file(filename, batch_size=2, sequence_len=3, validation_split=0.3)

