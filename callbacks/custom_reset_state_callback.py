from tensorflow import keras


class CustomResetStateCallback(keras.callbacks.Callback):
    def on_epoch_start(self, epoch, logs=None):
        print("Resetting model states")
        self.model.reset_states()
