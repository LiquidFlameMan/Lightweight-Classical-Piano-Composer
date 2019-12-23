import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.engine import InputLayer
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Activation, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow import keras


def train_network():
    # Получение массива нот
    notes = get_notes()
    # Получение количества уникальных нот
    n_vocab = len(set(notes))
    # Подготовка входных и выходных последовательностей
    network_input, network_output = prepare_sequences(notes, n_vocab)
    # Создание модели
    model = create_network(network_input, n_vocab)
    # Обучение модели
    train(model, network_input, network_output)


def get_notes():
    notes = []
    # Чтение всех MIDI файлов из midi_songs_new
    for file in glob.glob("midi_songs_new/*.mid"):
        # Перевод нот в формат music21
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:
            notes_to_parse = midi.flat.notes
        # Перевод в массив строк
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    # Дамп в файл notes
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    # Длина последовательности, подаваемой на вход
    sequence_length = 100
    # Создание отсортированного множества уникальных нот
    pitchnames = sorted(set(item for item in notes))
    # Создание словаря из чисел и нот
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    # Вход, выход сети
    network_input = []
    network_output = []
    # Подготовка входных и выходных последовательностей
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    # Число последовательностей
    n_patterns = len(network_input)
    # Изменение размерности массива и его нормализация
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    # Перевод выходного массива в бинарную матрицу
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    # Создание модели
    model = Sequential()
    # Вход нейросети
    model.add(InputLayer(input_shape=(network_input.shape[1], network_input.shape[2]),))
    # 3 GRU слоя
    model.add(GRU(
        128,
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(GRU(128, return_sequences=True, recurrent_dropout=0.3,))
    model.add(GRU(128))
    # Слой используется для уменьшения вероятности повторения одной ноты
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Полносвязный слой c полулинейной функцией активации
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.2))
    # Полносвязный слой с числом слоёв равным числу уникальных нот
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    # Название промежуточных файлов весов
    filepath = "weight_new/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')

    model.fit(network_input, network_output, epochs=200, batch_size=256, callbacks=[checkpoint])


if __name__ == '__main__':
    train_network()
