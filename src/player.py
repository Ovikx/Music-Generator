import model
import time

def read_sequence(file_name):
    with open(file_name) as f:
        raw = [seq.split(' ') for seq in f.readlines()]
        print(raw)
        for row, seq in enumerate(raw):
            for col, note in enumerate(seq):
                if '\n' in note[-1]:
                    print('edited')
                    raw[row][col] = note[:-1]
        return raw

gen = model.Generator()
seqs = read_sequence('generated_songs/generated_sequence250.txt')
for seq in seqs:
    print('Starting new sequence...')
    gen.play_sequence(seq, 100)
    time.sleep(3)
