from music21 import *
import glob

def load_lavender():
    lavenders = []
    lavender_score = converter.parse(
        '/Users/jesse/Documents/Code/AMLI/Projects/Final/pokemon_lavender.mid')
    lavenders.append(lavender_score)
    return lavenders

def load_midi_files_from(dir):
    """ dir: folder name with midi files.
        Returns a list of Score objects (music21) from the given directory. """
    lavenders = []
    for file in glob.glob(dir + '/*.mid'):
        lavender_score = converter.parse(file)
        lavenders.append(lavender_score)
    return lavenders
