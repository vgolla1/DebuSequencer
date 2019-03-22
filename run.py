from music21 import *
from load_midi import *
import midi_processing
import model_processing

import pandas as pd

def main():
    s = load_lavender()
    t = load_midi_files_from('/Users/jesse/Documents/Code/AMLI/Projects/Final')

    s_seq = midi_processing.get_all_sequences(s)
    s_data = model_processing.seq_to_data(s_seq)

    t_seqs = []
    for seq in t:
        t_seqs.append(midi_processing.get_all_sequences(seq))

    t_data = []
    for seq in t_seqs:
        t_data.append(model_processing.seq_to_data(seq))

    X, y, vocab = model_processing.get_X_y_vocab(s_data)
    rnn = model_processing.get_model(X, vocab)
    rnn_trained = model_processing.train_model(rnn, X, y)
    print(rnn_trained)

if __name__ == "__main__": main()
