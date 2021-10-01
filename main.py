import numpy as np
import matplotlib.pyplot as plt
import progressbar
import time
import gc

from scipy.io.wavfile import read, write
from keras.models import Model, load_model
from keras.layers import LSTM, LeakyReLU, Input, RepeatVector, TimeDistributed, Dense
# from keras.callbacks import ModelCheckpoint, EarlyStopping

model_path = 'Model\\MusicAI.h5'  # Path to save the AI model
sequence = 64  # Number of data AI look at to predict
sequence_loop = 3  # number of loop for each encoder and decoder
layer = 192  # Number of layers

validation_split = 0.8  # Ratio of data that is used for training
patience = 5  # Number of epoch to wait before terminate training the model
batch_size = 32  # Size of batch
epoch = 2  # number of epoch
training_step = 2  # Number of steps used for training before new epoch
default_rate = 44100  # Default rate for .wav file
sample_rate = 441  # The rate used for getting data from wav file (100 times lower than default rate)

precision = 'int16'  # Type of precision of data
max_precision = 32767  # The maximum number in the sound data
min_precision = -32768  # The minimum number in the sound data

num_song = 20  # Number of song in the data
debug = True  # Display all necessary information if true


def load_data(song_index):
    x_train, y_train, x_test, y_test = [], [], [], []
    widgets = ['Loading Data: ', progressbar.Bar()]

    # Read the .wav file and eliminate the non-essential
    rate, music = read('Data\\tobu.wav')
    music = music[int(len(music) * 0.1): int(len(music) * 0.9)]
    index = int(len(music) / (20 - song_index))

    bar = progressbar.ProgressBar(maxval=len(music) - sequence - 1, widgets=widgets)
    bar.start()

    # Load data
    for i in range(0, int(len(music) / 20), int(default_rate / sample_rate)):
        bar.update(i)

        # Hot encode label
        label = np.zeros((2, int(max_precision - min_precision)), dtype=precision)
        label[0][music[index + i + sequence][0] - min_precision - 1] = 1
        label[1][music[index + i + sequence][1] - min_precision - 1] = 1
        # print(str('[' + str(np.argmax(label[0]) + min_precision)) + ' ' + str(np.argmax(label[1]) + min_precision) + ']')
        # print(str('[' + str(np.argmin(label[0]) - max_precision)) + ' ' + str(np.argmin(label[1]) - max_precision) + ']')

        if i < len(music) / 20 * validation_split:
            x_train.append(music[index + i: index + i + sequence, :])
            y_train.append(label)
        else:
            x_test.append(music[index + i: index + i + sequence, :])
            y_test.append(label)
    bar.finish()

    # Convert data into single precision (FP32)
    x_train = np.asarray(x_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_train = np.asarray(y_train).astype('float16')
    y_test = np.asarray(y_test).astype('float16')

    # Normalize data
    x_train = (x_train - min_precision) / (max_precision - min_precision)
    x_test = (x_test - min_precision) / (max_precision - min_precision)

    if debug:
        print(np.shape(x_train))
        print(np.shape(y_train))
        print(np.shape(x_test))
        print(np.shape(y_test))

    return x_train, y_train, x_test, y_test


def encoder_decoder_lstm():
    # Input node of the neural network for 2 channels
    input_x = Input(shape=(sequence, 2))
    x = LSTM(layer, return_sequences=True)(input_x)

    # Encoder
    for i in range(sequence_loop):
        x = LSTM(int(layer / (2 ** (i + 1))), return_sequences=True)(x)
        x = LSTM(int(layer / (2 ** (i + 1))), return_sequences=True)(x)
        x = LeakyReLU()(x)

    # Hidden state
    x = LSTM(int(layer / (2 ** (sequence_loop + 1))))(x)
    x = RepeatVector(sequence)(x)

    # Decoder
    for i in range(sequence_loop):
        x = LSTM(int(layer / (2 ** (3 - i))), return_sequences=True)(x)
        x = LSTM(int(layer / (2 ** (3 - i))), return_sequences=True)(x)
        x = LeakyReLU()(x)

    x = LSTM(4)(x)
    x = RepeatVector(2)(x)

    # output
    output_x = TimeDistributed(Dense(max_precision - min_precision, activation='softmax'))(x)

    model = Model(inputs=input_x, outputs=output_x)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

    if debug:
        print(model.summary())

    return model


def plot_history(loss, acc):
    fig = plt.figure()

    result = fig.add_subplot(1, 1, 1)
    result.plot(loss, label='loss')
    result.plot(acc, label='acc')

    plt.savefig(model_path.split('h5')[0] + '.png')


def train():
    # Build model
    model = encoder_decoder_lstm()
    loss, acc = [], []

    # Start training
    for i in range(epoch):
        print('Epoch: ' + str(i + 1))

        for j in range(num_song - 1):
            # Load data and train model
            x_train, y_train, x_test, y_test = load_data(j)

            widgets = ['Training: ', progressbar.Bar()]
            bar = progressbar.ProgressBar(maxval=(len(x_train) + len(x_test)) * training_step, widgets=widgets)
            bar.start()

            for steps in range(training_step):
                for k in range(0, len(x_train), batch_size):
                    model.train_on_batch(x=x_train[k: k + batch_size], y=y_train[k: k + batch_size], reset_metrics=False)
                    bar.update(k + (len(x_train) + len(x_test)) * steps)

                for k in range(0, len(x_test), batch_size):
                    model_result = model.test_on_batch(x=x_test[k: k + batch_size], y=y_test[k: k + batch_size], reset_metrics=False, return_dict=True)
                    loss.append(model_result['loss'])
                    acc.append(model_result['acc'])
                    bar.update(k + len(x_train) + (len(x_train) + len(x_test)) * steps)

            model.save(model_path)
            plot_history(loss, acc)
            bar.finish()
            # model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=training_step, callbacks=[ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'), EarlyStopping(monitor='val_loss', mode='min', patience=patience)])

            # Free the memory to prevent out of memory
            time.sleep(10)
            model.reset_metrics()
            gc.collect()


def compose_music(seed=np.random.randint(0, 99999), seconds=5):
    # Set seed
    np.random.seed(seed)

    # Generate data and music
    data = np.random.randint(np.random.randint(min_precision, -1), np.random.randint(0, max_precision), (1, sequence, 2))
    music = np.copy(data)
    data = (data - min_precision) / (max_precision - min_precision)

    # Load model
    model = load_model(model_path)

    # This is used for visualization
    widgets = ['Generating Music: ', progressbar.Bar()]
    bar = progressbar.ProgressBar(maxval=sample_rate * seconds, widgets=widgets)
    bar.start()

    for i in range(sample_rate * seconds):
        bar.update(i)

        # Interference
        result = model.predict(data)
        print(np.shape(result[0]))
        print(np.argmax(result[0][0]))
        print(np.argmax(result[0][1]))
        print(result[0][0])
        input('wait1')
        final_result = [int(np.argmax(result[0][0]) - max_precision), int(np.argmax(result[0][1]) - max_precision)]
        final_result = np.asarray(final_result)
        final_result = np.expand_dims(final_result, axis=0)
        final_result = np.expand_dims(final_result, axis=0)
        model.reset_states()
        model.reset_metrics()

        # Add result to music
        music = np.concatenate((music, final_result), axis=1)

        # Process data for the next interference
        next_data = (final_result - min_precision) / (max_precision - min_precision)
        data = data[:, len(next_data):, :]
        data = np.concatenate((data, next_data), axis=1)

    bar.finish()

    if debug:
        print(np.shape(music[0]))

    music = music.astype(precision)
    write('Generated_Music.wav', sample_rate, music[0])


def main():
    print("Enter 1 to train the neural network, 2 to compose music")

    while True:
        user = input('Enter: ')
        flag = True

        if user == '1':
            train()
        elif user == '2':
            user_seed = input('Enter random number to compose different music or leave blank for random: ')

            if user_seed == '' or user_seed.isdigit():
                if user_seed == '':
                    user_seed = np.random.randint(0, 99999)

                user_time = input('Enter the number of seconds of music to be generated or leave blank for default 10 seconds: ')

                if user_time == '':
                    user_time = 10

                if user_time.isdigit():
                    compose_music(user_seed, user_time)
            else:
                print('Invalid choice')
                flag = False
        else:
            print('Invalid choice')
            flag = False

        if flag:
            break


if __name__ == '__main__':
    compose_music()
    input('wait')

    a, b = read('Generated_Music.wav')
    for i in range(len(b)):
        print(b[i])

    input('wait')
    compose_music()
