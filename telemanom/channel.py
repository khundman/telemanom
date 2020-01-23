import numpy as np
import os
import logging

logger = logging.getLogger('telemanom')


class Channel:
    def __init__(self, config, chan_id):
        """
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        self.config = config
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        self.train = None
        self.test = None

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM. config.l_s specifies
        the sequence length of prior timesteps fed into the model at
        each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """
        n_windows = len(arr) - self.config.l_s - self.config.n_predictions + 1

        x = np.lib.stride_tricks.as_strided(
            arr[:-self.config.n_predictions],
            shape=(n_windows, self.config.l_s, arr.shape[-1]),
            strides=(arr.strides[0], arr.strides[0], arr.strides[1])
        )

        y = np.lib.stride_tricks.as_strided(
            arr[self.config.l_s:, 0],  # telemetry value is at position 0
            shape=(n_windows, self.config.n_predictions),
            strides=(arr.strides[0], arr.strides[0])
        )

        if train:
            permutation = np.random.permutation(n_windows)
            self.X_train = x[permutation]
            self.y_train = y[permutation]
        else:
            self.X_test = x
            self.y_test = y

    def load_data(self):
        """
        Load train and test data from local.
        """
        try:
            self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)