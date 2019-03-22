import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
from music21 import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.utils import np_utils
from random import sample

SEQ_LENGTH = 3

TEMPOS = {
    (0,40):0,
    (40,60):1,
    (60,80):2,
    (80,120):3,
    (120,170):4,
    (170,400):5
}

def seq_to_data(sequences):
    """ sequences: A sequence of music events (notes, chords, rests)

        Converts music events to a numpy array of data points containing pitch information
        (grouped in a tuple if there is more than a single pitch), velocity, and duration
        about the current note or chord, and other general information about the current meter.

        Returns seq, conversions: the sequence of data points converted into categorical data
        and a list of conversions to translate data back into note information. """
    seq_full = []
    for i in range(len(sequences)):
        current_seq = sequences[i] # [(pitch(es), vel, dur), inst, tempo, key, time_sig]

        # Unpack all additional sequence information
        current_inst = current_seq[1]
        current_tempo = current_seq[2]
        current_key = current_seq[3]
        current_time_sig = current_seq[4]

        # Unpack note/chord data and formulate into data points
        pitch_data = current_seq[0]
        for j in range(len(pitch_data)):
            # Grab information specific to the note or chord we're looking at
            pitches = pitch_data[j][0]
            velocity = pitch_data[j][1]
            duration = pitch_data[j][2]

            # Combine pitches, tempo, key, time sig, velocity, duration into a single data point
            data_point = (
                pitches,
                current_tempo,
                current_key,
                current_time_sig,
                velocity,
                duration
            )

            seq_full.append(data_point)

    return seq_full

def get_X_y_vocab(data):
    """ data: a list of lists that is complete note sequences of all training pieces
        Returns:
        X:      normalized sequences of length SEQ_LENGTH to use as input to the model
        xprime: un-normalized sequences of lenght SEQ_LENGTH to use for predicting new music
        y:      single value sequences representing the labels of the input data to the model
        vocab:  the count of unique notes found throughout all training music,
                useful for the output layer of a neural network to know
                the number of output classes """
    flat_data = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            flat_data.append(data[i][j])
    _, factors = pd.factorize(flat_data)
    vocab = len(factors)

    cat_data = []
    for i in range(len(data)):
        cat_data.append([])
        for j in range(len(data[i])):
            idx = list(factors).index(data[i][j])
            cat_data[i].append(idx)

    seq_in, seq_out = [], []
    for i in range(len(cat_data)):
        current_seq = cat_data[i]
        for j in range(len(current_seq) - SEQ_LENGTH):
            seq_in.append(current_seq[j:j+SEQ_LENGTH])
            seq_out.append(current_seq[j+SEQ_LENGTH])

    # Should we convert the seq_in to a tuple instead of a list?
    X, y = array(seq_in), array(seq_out)
    X = X.reshape(1, len(X), len(X[0]))
    y = y.reshape(1, len(y), 1)

    return X, y, vocab

def get_model(X, vocab, lstm_nodes=512, dense_nodes=256, dropout_rate=0.3):
    model = Sequential()
    model.add(LSTM(
        lstm_nodes,
        name='DebuSequencer',
        input_shape=(X.shape[1], X.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_nodes, name='Bach_size', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_nodes, name='Nodezart', return_sequences=True))
    model.add(Dense(dense_nodes))
    model.add(Dropout(dropout_rate))
    model.add(Dense(vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    return model

def train_model(model, X, y, epochs=10):
    return model.fit(X, y, epochs=epochs, shuffle=False, verbose=0)
