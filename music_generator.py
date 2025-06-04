import sys
import re
import numpy as np
import pandas as pd
from music21 import converter, instrument, note, chord, stream
from glob import glob
import IPython
from tqdm import tqdm
import pickle
import os
import datetime

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

os.makedirs('Jazz', exist_ok=True)
os.makedirs('Music Generation', exist_ok=True)
os.makedirs('data', exist_ok=True)


songs = glob('Jazz/*.mid')
if not songs:
    print("Warning: No MIDI files found in the 'Jazz' directory. Please add some MIDI files to continue.")
    sys.exit(1)


def get_notes():
    notes = []
    for file in songs:
        try:
            midi = converter.parse(file)
            notes_to_parse = []
            try:
                parts = instrument.partitionByInstrument(midi)
            except:
                pass
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    if not notes:
        print("Error: No notes were extracted from the MIDI files.")
        sys.exit(1)
    if len(notes) < 101:
        print(f"Error: Not enough notes extracted ({len(notes)} found). At least 101 notes are required for sequence generation.")
        sys.exit(1)
    with open('Music Generation/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    return notes

def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_in, n_vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def train(model, network_input, network_output, epochs):
    filepath = 'weights.best.music3.keras'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
    model.fit(network_input, network_output, epochs=epochs, batch_size=32, callbacks=[checkpoint])

def train_network():
    epochs = 200
    notes = get_notes()
    print('Notes processed')
    n_vocab = len(set(notes))
    print('Vocab generated')
    network_in, network_out = prepare_sequences(notes, n_vocab)
    if len(network_in) == 0:
        print("Error: Not enough data to create input sequences. Please add more or longer MIDI files.")
        sys.exit(1)
    print('Input and Output processed')
    model = create_network(network_in, n_vocab)
    print('Model created')
    train(model, network_in, network_out, epochs)
    print('Training completed')
    return model

def generate():
    try:
        with open('Music Generation/notes', 'rb') as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        print("Error: Notes file not found. Please run the training first.")
        sys.exit(1)
    if len(notes) < 101:
        print(f"Error: Not enough notes in notes file ({len(notes)} found). At least 101 notes are required for generation.")
        sys.exit(1)
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    print('Initiating music generation process.......')
    network_input = get_inputSequences(notes, pitchnames, n_vocab)
    normalized_input = network_input / float(n_vocab)
    model = create_network(normalized_input, n_vocab)
    print('Loading Model weights.....')
    model.load_weights('weights.best.music3.keras')
    print('Model Loaded')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def get_inputSequences(notes, pitchnames, n_vocab):
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    network_input = np.reshape(network_input, (len(network_input), 100, 1))
    return network_input

def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = list(network_input[start])
    prediction_output = []

    print('Generating notes........')

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append([index])
        pattern = pattern[1:]

    print('Notes Generated...')
    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    # Generate a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'generated_{timestamp}.mid'
    print(f'Saving Output file as midi.... {filename}')
    midi_stream.write('midi', fp=filename)
    try:
        from music21 import midi
        player = midi.realtime.StreamPlayer(midi_stream)
        print('Playing generated MIDI...')
        player.play()
    except Exception as e:
        print(f'Could not play MIDI automatically. Please open {filename} manually. Error:', e)
    print(f"Done! Check the generated MIDI file: {filename}")

if __name__ == "__main__":
    train_network()
