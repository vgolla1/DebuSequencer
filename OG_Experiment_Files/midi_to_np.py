import tensorflow as tf
import keras
from keras.layers import RNN
import numpy as np
import pandas as pd
from music21 import *
from fractions import Fraction

INSTRUMENT = instrument.Instrument
METRONOME = tempo.MetronomeMark
KEY = key.Key
TIME_SIGNATURE = meter.TimeSignature

TEMPOS = {(0,40):0,
          (40,60):1,
          (60,80):2,
          (80,120):3,
          (120,170):4,
          (170,400):5}

lavender_score = converter.parse(
    '/Users/jesse/Documents/Code/AMLI/Projects/Final/pokemon_lavender.mid'
)
# s = instrument.partitionByInstrument(lavender_score)
# lavender_recurse = s.parts[0].recurse()
# lavender_recurse.show()

lavender_recurse = lavender_score.recurse()

# for i in range(len(lavender_score.parts)):
#     lavender_score.parts[i].show()

def extract_instrument(element):
    return element.instrumentName

def extract_tempo_mark(element):
    return str(element.text), str(element.number)

def extract_key(element):
    #return str(element.tonic) + str(element.mode)
    return element.sharps

def extract_time_sig(element):
    return element.ratioString

def extract_quarter_length(q_length):
    if isinstance(q_length, Fraction):
        return float("{0:.3f}".format(q_length.numerator / q_length.denominator))
    return q_length

def extract_rest(element):
    return '&&', '0', extract_quarter_length(element.duration.quarterLength)#, element.offset

def extract_note(element):
    return (str(element.pitch.name) +
            str(element.pitch.octave),
            str(element.volume.velocity),
            extract_quarter_length(element.quarterLength)#, element.offset
            )

def extract_chord(element):
    current_chord = []
    for i in range(len(element.pitches)):
        current_chord.append(str(element.pitches[i].name) + str(element.pitches[i].octave))
    current_chord = [tuple(current_chord)]
    current_chord.append(element.volume.velocity),
    current_chord.append(extract_quarter_length(element.duration.quarterLength))
    return current_chord

def save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig):
    sequence_to_add = [current_sequence]
    sequence_to_add.append(current_inst)
    sequence_to_add.append(current_metro)
    sequence_to_add.append(current_key)
    sequence_to_add.append(current_time_sig)
    sequences.append(sequence_to_add)
    return sequences

sequences = []
current_sequence = []
current_inst, current_metro, current_key, current_time_sig = None, None, None, None
new_sequence = False
for element in lavender_recurse:
    # Check is we have new measure information (instrument, tempo, key, time_sig)
    if (isinstance(element, INSTRUMENT) or
        isinstance(element, METRONOME)  or
        isinstance(element, KEY)        or
        isinstance(element, TIME_SIGNATURE)):
    # If yes, we should try to save the previous set of sequences and their measure information AND reset the current sequence
        if new_sequence:
            sequences = save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig)
            new_sequence = False
            current_sequence = []
    # Also save this new measure information over the old information
        if isinstance(element, INSTRUMENT):
            current_inst = extract_instrument(element)
        if isinstance(element, METRONOME):
            current_metro = extract_tempo_mark(element)
        if isinstance(element, KEY):
            current_key = extract_key(element)
        if isinstance(element, TIME_SIGNATURE):
            current_time_sig = extract_time_sig(element)
    # Otherwise we have note/chord data and that should be added to the current sequence
    if isinstance(element, note.Rest):
        if not new_sequence:
            new_sequence = True
        current_sequence.append(extract_rest(element))
    if isinstance(element, note.Note):
        if not new_sequence:
            new_sequence = True
        current_sequence.append(extract_note(element))
    if isinstance(element, chord.Chord):
        if not new_sequence:
            new_sequence = True
        current_sequence.append(extract_chord(element))
# We will have to save the last sequence... perhaps outside of the loop
save_sequence(sequences, current_sequence, current_inst, current_metro, current_key, current_time_sig)

# Look and see what we have! This would be a place to convert this data for the RNN/GAN down the road
print()
for i in range(len(sequences)):
    for j in range(len(sequences[i])):
        print(type(np.array(sequences[i])[j]))
        print(np.array(sequences[i])[j])
        print()

print(np.array(sequences).shape)
print()

# -------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array

seq_full = []
for i in range(len(sequences)):
    current_tempo = None
    if sequences[i][2]:
        current_tempo = int(float(sequences[i][2][1]))
    for r in TEMPOS.keys():
        if current_tempo in range(r[0], r[1]):
            current_tempo = TEMPOS[r]
            break

    current_key = sequences[i][3]
    current_time_sig = sequences[i][4]
    for j in range(len(sequences[i][0])):
        if len(sequences[i][0]) > 2:    # We have a chord to unpack
            seq_full.append((current_tempo, current_key, current_time_sig, *sequences[i][0][j][:-2], sequences[i][0][j][-2], sequences[i][0][j][-1])) # Grab the tempo, key, time sig, all pitches, velocity, duration
        else:   # Just a note
            seq_full.append((current_tempo, current_key, current_time_sig, sequences[i][0][j][0], sequences[i][0][j][1], sequences[i][0][j][2]))  # Grab the tempo, key, time sig, pitch, velocity, duration

# print(seq_full)
# print()

seq_full, conversions = pd.factorize(seq_full)

# print('seq_full')
# print(seq_full)
# print()
# print('conversions')
# print(conversions)
# print()

seq = []
for i in range(len(seq_full) - 1):
    seq.append([seq_full[i], seq_full[i+1]])

seq = array(seq)
X, y = seq[:, 0], seq[:, 1]
X = X.reshape((len(X), 1, 1))

# print('seq')
# print(seq)

model = Sequential()
model.add(LSTM(10, input_shape=(1,1)))
model.add(Dense(1, activation='linear'))    #linear -> softmax

model.compile(loss='mse', optimizer='adam') #sparse cross-entropy?

model.fit(X, y, epochs=300, shuffle=False, verbose=0)
print()

y_hat = model.predict(X, verbose=0)
for i in range(len(y_hat)):
    y_hat[i] = np.round(y_hat[i])

p_notes = []
for j in range(len(y_hat)):
    p_notes.append(conversions[int(y_hat[j])])

# print('p_notes')
# print(p_notes)

# --------

output_notes = []
overall_offset = 0.0
for i in range(len(p_notes)):
    if p_notes[i][3] == '&&':
        new_note = note.Rest()
    else:
        if isinstance(p_notes[i][3], tuple):
            pitches = []
            for pitch in p_notes[i][3]:
                pitches.append(pitch)
            new_note = chord.Chord(pitches)
        else:
            new_note = note.Note(p_notes[i][3])

    new_note.volume = volume.Volume(velocity=int(p_notes[i][4]))
    new_note.duration = duration.Duration(p_notes[i][5])
    new_note.offset = overall_offset
    overall_offset += new_note.duration.quarterLength
    new_note.storedInstrument = instrument.Piano()
    output_notes.append(new_note)

#--- Output stuff, uncomment to save to file and check out generated music
midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='peepthis.mid')
peep = converter.parse('./peepthis.mid').show()
