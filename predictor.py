import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Dropout, Activation

def generate_music():
    # Generate a piano MIDI file
    # Load the notes used to train the model
    with open('data/notes', 'rb') as file:
        training_notes = pickle.load(file)

    # Get unique pitch names
    pitch_names = sorted(set(item for item in training_notes))
    # Get the number of unique pitches
    num_pitches = len(set(training_notes))

    network_input, normalized_input = prepare_sequences(training_notes, pitch_names, num_pitches)
    model = create_network(normalized_input, num_pitches)
    prediction_output = generate_notes(model, network_input, pitch_names, num_pitches)
    create_midi(prediction_output)

def prepare_sequences(training_notes, pitch_names, num_pitches):
    # Prepare the sequences used by the Neural Network
    # Map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(training_notes) - sequence_length, 1):
        in_sequence = training_notes[i:i + sequence_length]
        out_sequence = training_notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in in_sequence])
        output.append(note_to_int[out_sequence])

    num_patterns = len(network_input)

    # Reshape the input for compatibility with LSTM layers
    normalized_input = np.reshape(network_input, (num_patterns, sequence_length, 1))
    # Normalize input
    normalized_input = normalized_input / float(num_pitches)

    return network_input, normalized_input

def create_network(normalized_input, num_pitches):
    # Create the structure of the neural network
    model = Sequential()
    model.add(LSTM(512, input_shape=(normalized_input.shape[1], normalized_input.shape[2]), recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load weights to each node
    model.load_weights('weights.hdf5')

    return model

def generate_notes(model, network_input, pitch_names, num_pitches):
    # Generate notes from the neural network based on a sequence of notes
    # Pick a random sequence from the input as a starting point for the prediction
    start_index = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    pattern = network_input[start_index]
    prediction_output = []

    # Generate 500 notes
    for _ in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(num_pitches)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    # Convert the output from the prediction to notes and create a MIDI file from the notes
    offset = 0
    output_notes = []

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = [note.Note(int(current_note)) for current_note in notes_in_chord]
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='generated_music.mid')

if __name__ == '__main__':
    generate_music()
