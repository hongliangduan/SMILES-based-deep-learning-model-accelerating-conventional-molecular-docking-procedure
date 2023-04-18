from typing import Callable, Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar
from functools import partial

import tensorflow as tf
import numpy as np
from numpy import ndarray
from tensorflow import keras

from featurizer import Featurizer, feature_matrix
import re
from tqdm import tqdm

T = TypeVar("T")
T_feat = TypeVar("T_feat")
Dataset = tf.data.Dataset

def mve_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    var = tf.math.softplus(y_pred[:, 1])

    return tf.reduce_mean(
        tf.math.log(2 * np.pi) / 2 + tf.math.log(var) / 2 + (mu - y_true) ** 2 / (2 * var)
    )

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class FinLSTM:
    def __init__(
        self,
        input_size: int,
        batch_size: int = 128,
        layer_sizes: Optional[Sequence[int]] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = "relu",
        uncertainty: Optional[str] = None,
        model_seed: Optional[int] = None,
        dropout_size: int = 10,
        ):

        self.input_size = input_size
        self.batch_size = batch_size

        self.uncertainty = uncertainty
        self.dropout_size = dropout_size

        layer_sizes = layer_sizes or [100, 100]
        self.model, self.optimizer, self.loss = self.build(
            input_size, 1, layer_sizes, dropout, self.uncertainty, activation
        )

        self.mean = 0
        self.std = 0

        tf.random.set_seed(model_seed)
    
    def build(self, input_size, num_tasks, layer_sizes, dropout, uncertainty, activation):
        dropout_at_predict = uncertainty == "dropout"
        output_size = 2*num_tasks if self.uncertainty else num_tasks

        inputs = keras.layers.Input(shape=(input_size,))
        embedding = keras.layers.Embedding(output_dim=layer_sizes[0], input_dim=input_size)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(layer_sizes[0], return_sequences=True), merge_mode='concat')(embedding)
        hidden = lstm

        for layer_size in layer_sizes:
            hidden = keras.layers.Dense(
                units=layer_size,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(0.05),
            )(hidden)

            if dropout:
                hidden = keras.layers.Dropout(dropout)(hidden, training=dropout_at_predict)
        
        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(output_size, activation="linear")(hidden)

        model = keras.Model(inputs, outputs)

        if uncertainty not in {"mve"}:
                optimizer = keras.optimizers.Adam(lr=0.01)
                loss = keras.losses.mse
        elif uncertainty == "mve":
                optimizer = keras.optimizers.Adam(lr=0.01)
                loss = mve_loss
        else:
            raise ValueError(f'Unrecognized uncertainty method: "{uncertainty}"')

        return model, optimizer, loss

    def train(
        self, xs: Iterable[T], ys: Iterable[float], featurizer: Callable[[T], ndarray]
    ):

        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.summary()

        X = np.array(feature_matrix(xs, featurizer))
        print(X.shape)
        Y = self._normalize(ys)

        self.model.fit(
            X,
            Y,
            batch_size=self.batch_size,
            validation_split=0.2,
            epochs=100,
            validation_freq=2,
            verbose=1,
        )
    
        return True

    def predict(self, xs: Sequence[ndarray]):
        X = np.stack(xs, axis=0)
        Y_pred = self.model.predict(X)

        if self.uncertainty == "mve":
            Y_pred[:, 0::2] = Y_pred[:, 0::2] * self.std + self.mean
            Y_pred[:, 1::2] = Y_pred[:, 1::2] * self.std**2
        else:
            Y_pred = Y_pred * self.std + self.mean

        return Y_pred

    def _normalize(self, ys: Sequence[float]) -> ndarray:
        Y = np.stack(list(ys))
        self.mean = np.nanmean(Y, axis=0)
        self.std = np.nanstd(Y, axis=0)

        return (Y - self.mean) / self.std

class SmiLSTM:
    def __init__(
        self,
        batch_size: int = 128,
        layer_sizes: Optional[Sequence[int]] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = "relu",
        model_seed: Optional[int] = None,
        ):

        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=None, 
                    filters='\t\n',
                    lower=False,
                    split=' '
        )

        # self.tokenizer.fit_on_texts(xs)
        # self.vocab_size = self.tokenizer.texts_to_matrix(xs).shape[-1]
        # self.xs = self.tokenizer.texts_to_sequences(xs)
        # self.input_size = len(max(self.xs, key=len))
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.activation = activation

        tf.random.set_seed(model_seed)

    def build(self, input_size, layer_sizes, dropout, activation):

        inputs = keras.layers.Input(shape=(input_size,))
        embedding = keras.layers.Embedding(output_dim=layer_sizes[0], input_dim=input_size)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(layer_sizes[0], return_sequences=True), merge_mode='concat')(embedding)
        hidden = lstm

        for layer_size in layer_sizes:
            hidden = keras.layers.Dense(
                units=layer_size,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(0.05),
            )(hidden)

            if dropout:
                hidden = keras.layers.Dropout(dropout)(hidden)
        
        hidden = keras.layers.Flatten()(hidden)
        outputs = keras.layers.Dense(1, activation="linear")(hidden)

        model = keras.Model(inputs, outputs)
        optimizer = keras.optimizers.Adam(lr=0.01)
        loss = keras.losses.mse

        return model, optimizer, loss

    def get_model(self, xs):
        
        self.tokenizer.fit_on_texts(xs)
        self.vocab_size = self.tokenizer.texts_to_matrix(xs).shape[-1]
        self.xs = self.tokenizer.texts_to_sequences(xs)
        self.input_size = len(max(self.xs, key=len))

        layer_sizes = self.layer_sizes or [100, 100]
        self.model, self.optimizer, self.loss = self.build(
            self.input_size, layer_sizes, self.dropout, self.activation
        )

    def train(
        self, ys,
    ):

        self.xs = keras.preprocessing.sequence.pad_sequences(self.xs, maxlen=self.input_size, padding='post')

        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.model.summary()

        self.model.fit(
            self.xs,
            ys,
            batch_size=self.batch_size,
            validation_split=0.2,
            epochs=100,
            validation_freq=2,
            verbose=1,
        )
    
        return True
    
    def predict(self, xs: Sequence[ndarray]):
        # X = np.stack(xs, axis=0)
        xss = []
        for x in xs:
            xss.append(smi_tokenizer(x))
        xss = self.tokenizer.texts_to_sequences(xss)
        xss = keras.preprocessing.sequence.pad_sequences(xss, maxlen=self.input_size, padding='post')
        Y_pred = self.model.predict(xss)

        return Y_pred

class LSTMDropoutModel():
    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 8192,
        dropout: Optional[float] = 0.2,
        dropout_size: int = 10,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 8192

        self.build_model = partial(
            FinLSTM,
            input_size=input_size,
            num_tasks=1,
            batch_size=test_batch_size,
            dropout=dropout,
            uncertainty="dropout",
            model_seed=model_seed,
        )
        self.model = self.build_model()
        self.dropout_size = dropout_size

    