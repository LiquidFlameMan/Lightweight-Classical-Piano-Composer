import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.utils import plot_model
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Activation, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow import keras


def train_network():
    notes = get_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    plot_model(model, to_file='model.png')

    train(model, network_input, network_output)


def get_notes():
    notes = []

    for file in glob.glob("midi_songs_new/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100

    pitchnames = sorted(set(item for item in notes))

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(InputLayer(input_shape=(network_input.shape[1], network_input.shape[2]),))
    model.add(GRU(
        128,
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(GRU(128, return_sequences=True, recurrent_dropout=0.3,))
    model.add(GRU(128))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=[keras.metrics.CategoricalAccuracy()])

    model.load_weights('weight_new/weights-improvement-107-0.5131-bigger.hdf5')

    return model


def train(model, network_input, network_output):
    filepath = "weight_new/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    model.fit(network_input, network_output, epochs=200, batch_size=256, callbacks=[checkpoint])


if __name__ == '__main__':
    train_network()
