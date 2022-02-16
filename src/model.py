import torch
import torch.nn as nn
from random import randint
import numpy as np
import winsound as ws
from mingus.midi import fluidsynth
from mingus.containers.note import Note
import time
import os

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the SoundFont file
fluidsynth.init('sf2/FluidR3_GM.sf2')

data = []
for file_name in os.listdir('songs'):
    with open(f'songs/{file_name}') as f:
        data.append(f.read())

data = ' \n '.join(data)

# Split the data by spaces
data = data.split(' ')
print(len(data))

# Define the notes
base = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Create all possible notes
all_notes = [f'{note}{octave}' for octave in range(1, 8) for note in base]

# Add miscellaneous items
all_notes.append('REST')
all_notes.append('\n')
for i in range(1, 4):
    all_notes.append(f'{i}x')
for i in range(300):
    all_notes.append(str(i))

n_notes = len(all_notes)
print(data)

class RNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(
            num_embeddings=embedding_size,
            embedding_dim=hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.dense = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )
    
    # Forward prop
    def forward(self, x, hidden, cell):
        output = self.embed(x)
        output, (hidden, cell) = self.lstm(output.unsqueeze(1), (hidden, cell))
        output = self.dense(output.reshape(output.shape[0], -1))
        return output, (hidden, cell)
    
    # Initializes the hidden/cell states
    def init_states(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        return hidden, cell


class Generator():
    def __init__(self):
        self.chunk_len = 100
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_interval = 50
        self.save_model_interval = 250
        self.save_sample_interval = 250
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.001

    # Converts a sequence string into a Torch tensor string
    def seq_to_tensor(self, seq):
        tensor = torch.zeros(len(seq)).long()

        for i, note in enumerate(seq):
            tensor[i] = all_notes.index(note)
        
        return tensor
    
    # Gets a random batch from the dataset
    def get_random_batch(self):
        start_idx = randint(0, len(data)-self.chunk_len-1)
        end_idx = start_idx + self.chunk_len + 1

        seq_sample = data[start_idx:end_idx]
        seq_input = torch.zeros(self.batch_size, self.chunk_len, device=device)
        seq_target = torch.zeros(self.batch_size, self.chunk_len, device=device)

        seq_input[0, :] = self.seq_to_tensor(seq_sample[:-1])
        seq_target[0, :] = self.seq_to_tensor(seq_sample[1:])
        
        return seq_input.long(), seq_target.long()
    
    # Generates samples
    def generate(self, initial_note='100', prediction_len = 100, temperature=1):
        initial_note = [initial_note]
        hidden, cell = self.rnn.init_states(batch_size=self.batch_size)
        initial_inp = self.seq_to_tensor(initial_note)
        predicted = initial_note

        for p in range(len(initial_note) - 1):
            _, (hidden, cell) = self.rnn(initial_inp[p].view(1).to(device), hidden, cell)
        
        last_char = initial_inp
        print(last_char.shape)
        for p in range(prediction_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.view(-1).div(temperature).exp()
            top_note = torch.multinomial(output_dist, 1)[0]
            predicted_note = all_notes[top_note]
            predicted += [predicted_note]
            last_char = self.seq_to_tensor([predicted_note])
        
        return predicted
    
    # Trains the RNN
    def train(self):
        self.rnn = RNN(
            embedding_size=n_notes,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=n_notes
        ).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        loss_function = nn.CrossEntropyLoss()
        print('Starting training...')

        for epoch in range(self.num_epochs):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_states(self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for i, c in enumerate(inp[0]):
                prediction, (hidden, cell) = self.rnn(c.unsqueeze(0), hidden, cell)
                loss += loss_function(prediction, target[0][i].unsqueeze(0))
            
            loss.backward()
            optimizer.step()

            loss = loss.item() / self.chunk_len

            if (epoch + 1) % self.print_interval == 0:
                print(f'Epoch {epoch+1} || Loss: {loss}')
                print(self.generate())
            
            if (epoch + 1) % self.save_sample_interval == 0:
                self.save_sequence(self.generate(), epoch+1)

    # Plays the sequence at a specified BPM
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
            if note == '\n':
                time.sleep(sleep_duration*3)
                print('Moving to next song')
                continue

            fluidsynth.play_Note(Note(note[:-1], int(note[-1]), channel=1, velocity=127))
            time.sleep(sleep_duration)
     
    # Saves the sequence into a text file
    def save_sequence(self, seq, epoch):
        compiled = ' '.join(seq).replace(' \n ', '\n')
        file_name = f'generated_songs/generated_sequence{epoch}.txt'
        with open(file_name, 'w') as f:
            f.write(compiled)
        print(f'Saved sequence to [{file_name}]')