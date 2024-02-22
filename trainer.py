import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_music_generator():
    # Train a Neural Network to generate music
    notes = get_notes()

    # Get the number of unique pitch names
    num_pitch_names = len(set(notes))

    network_input, network_output = prepare_sequences(notes, num_pitch_names)

    model = create_music_generator(network_input, num_pitch_names)

    train(model, network_input, network_output)

def get_notes():
    # Get all the notes and chords from the MIDI files in the ./midi_songs directory
    all_notes = []

    for file in glob.glob("midi_songs/*.mid"):
        midi_data = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # Check if the MIDI file has instrument parts
            instrument_partition = instrument.partitionByInstrument(midi_data)
            notes_to_parse = instrument_partition.parts[0].recurse()
        except:  # The file has notes in a flat structure
            notes_to_parse = midi_data.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                all_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                all_notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as file_path:
        pickle.dump(all_notes, file_path)

    return all_notes

def prepare_sequences(notes, num_pitch_names):
    # Prepare the sequences used by the Neural Network
    sequence_length = 100

    # Get all pitch names and create a mapping to integers
    pitch_names = sorted(set(item for item in notes))
    note_to_int_mapping = dict((note, number) for number, note in enumerate(pitch_names))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int_mapping[char] for char in sequence_in])
        network_output.append(note_to_int_mapping[sequence_out])

    num_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(num_pitch_names)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output

def create_music_generator(network_input, num_pitch_names):
    # Create the structure of the neural network
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]),
                   recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(num_pitch_names))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    # Train the neural network
    file_path = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_music_generator()
