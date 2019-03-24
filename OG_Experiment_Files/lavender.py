from music21 import *
import pandas as pd
import numpy as np
import tensorflow as tf
import magenta

INSTRUMENT = instrument.Instrument
METRONOME = tempo.MetronomeMark
KEY = key.Key
TIME_SIGNATURE = meter.TimeSignature

def extract_instrument(element):
    return element.instrumentName

def extract_tempo_mark(element):
    return str(element.text), str(element.number)

def extract_key(element):
    return str(element.tonic) + str(element.mode)

def extract_time_sig(element):
    return element.ratioString

def extract_rest(element):
    return '&&', element.duration.quarterLength, element.offset

def extract_note(element):
    return (str(element.pitch.name) +
            str(element.pitch.octave),
            element.quarterLength#, element.offset
            )

def extract_chord(element):
    current_chord = []
    for i in range(len(element.pitches)):
        current_chord.append(str(element.pitches[i].name) + str(element.pitches[i].octave))
    current_chord = [tuple(current_chord)]
    current_chord.append(element.duration.quarterLength)
    return current_chord

lavender_score = converter.parse(
    '/Users/jesse/Documents/Code/AMLI/Projects/Final/ff6shap.mid'
)
lavender_recurse = lavender_score.recurse()

for element in lavender_recurse:
    print('\n--- Current Element ---')
    print(type(element))
    if isinstance(element, INSTRUMENT):
        print(extract_instrument(element))
    elif isinstance(element, METRONOME):
        print(extract_tempo_mark(element))
    elif isinstance(element, KEY):
        print(extract_key(element))
    elif isinstance(element, TIME_SIGNATURE):
        print(extract_time_sig(element))
    elif isinstance(element, note.Rest):
        print(extract_rest(element))
    elif isinstance(element, note.Note):
        print(extract_note(element))
    elif isinstance(element, chord.Chord):
        print(extract_chord(element))
