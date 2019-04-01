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

#s.show()

a3 = note.Note('A3')
a4 = note.Note('A4')
f3 = note.Note('F3')
f4 = note.Note('F4')

a3.offset = 0.0
a4.offset = 1.0
f3.offset = 0.0
f4.offset = 1.0

stream1 = stream.Stream()
stream1.insert(a3)
stream1.insert(a4)

stream2 = stream.Stream()
stream2.insert(f3)
stream2.insert(f4)

st = stream.Stream()
st.insert(stream1)
st.insert(stream2)

st.show()
