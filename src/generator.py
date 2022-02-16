import numpy as np
import winsound as ws
from mingus.midi import fluidsynth
from mingus.containers.note import Note
import time

fluidsynth.init('sf2/FluidR3_GM.sf2')

with open('songs/My-Castle-Town.txt') as f:
    data = f.read()

data = data.split(' ')

class Generator():
    def __init__(self):
        self.base = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        self.notes = [f'{note}{octave}' for octave in range(1, 8) for note in self.base]

    def note_to_freq(self, note):
        return round(440*(2**(1/12))**((self.notes.index(note) - self.notes.index('C4')) - 9))
    
    def play_sequence(self, seq, bpm):
        last_mult = 1

        for note in seq:
            if note.isdigit():
                bpm = int(note)
                continue

            if 'x' in note:
                bpm = round(bpm*int(note[:-1])/last_mult)
                last_mult = int(note[:-1])
                continue
            
            sleep_duration = 60/bpm
            if note == 'REST':
                time.sleep(sleep_duration)
                continue
            fluidsynth.play_Note(Note(note[:-1], int(note[-1]), channel=1, velocity=127))
            time.sleep(sleep_duration)


gen = Generator()
print(gen.play_sequence(data, 100))