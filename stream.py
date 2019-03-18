from music21 import *

s = stream.Measure()

time_sig1 = meter.TimeSignature('3/4')

s.insert(0, time_sig1)
s.insert(0, key.KeySignature(2))
s.insert(0, clef.TrebleClef())
s.insert(0, note.Note('C#4'))
s.insert(1, note.Note('D#4'))

e = note.Note('e4')
s.append(e)

# insert will place a note in the stream AFTER an element (if it exists)
# len() of a stream will include 'spaceless' elements like time sig, key sig, etc.
s.insert(len(s) - 3, note.Note('b3')) # <-- places at current end 

s.show()
