from music21 import *
#converter.parse("tinynotation: 3/4 c4 g8 e d16 e d c").show()

ff6 = converter.parse('/Users/jesse/Documents/Code/AMLI/Projects/Final/pokemon_lavender.mid')
ff6.show()

# notes = []
#
# notes_to_parse = None
#
# try:
#     s2 = instrument.partitionByInstrument(ff6)
#     notes_to_parse = s2.parts[0].recurse()
# except:
#     notes_to_parse = ff6.flat.notes
#
# print([notes_to_parse[10].pitches[0].name + str(notes_to_parse[10].pitches[0].octave),
#        notes_to_parse[10].pitches[1].name + str(notes_to_parse[10].pitches[0].octave)])
# print(notes_to_parse[11].pitch)

# for part in ff6:
#     for element in part:
for element in ff6.recurse():
    if isinstance(element, note.Note):
        print(element.pitch, element.quarterLength)
    elif isinstance(element, chord.Chord):
        current_chord = []
        for i in range(len(element.pitches)):
            current_chord.append(str(element.pitches[i].name) + str(element.pitches[i].octave))
        current_chord = [tuple(current_chord)]
        current_chord.append(element.duration.quarterLength)
        print(current_chord)
    elif isinstance(element, note.Rest):
        print('&', element.duration.quarterLength)
    elif isinstance(element, tempo.MetronomeMark):
        print(element.text, element.number)
    elif isinstance(element, instrument.Instrument):
        print("Instrument:", element.instrumentName)
    elif isinstance(element, key.Key):
        print("Key:", element.tonic, element.mode)
    elif isinstance(element, meter.TimeSignature):
        print("Time Signature:", element.ratioString)
