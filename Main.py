from Logs import *
from Model import *
from GenerateAudio import *
from scipy import signal
from Monitoring import *

TRAINING = False
export_model = Model()

experiment = {
    'id_experiment': 1,
    'lstm_neurons': 100,
    'n_epochs': 1,
    'q_levels': 256,
    'lstm_activation': 'relu',
    'dense_activation': 'relu',
    'batch_size': 32,
    'number_of_layers': 1,
    'random': False,
    'train_set_shape': 2
}

# import datasets
ableton_train = np.load('../music-Ableton_new/ableton_train.npy')
ableton_valid = np.load('../music-Ableton_new/ableton_valid.npy')
ableton_test = np.load('../music-Ableton_new/ableton_test.npy')

# cut the notes
train_dataset = []
valid_dataset = []
test_dataset = []

sample_limit = 22050
notes_limit = experiment['train_set_shape']

for i_note in range(ableton_train.shape[0]):
    if i_note < notes_limit:
        if i_note < notes_limit:
            note = ableton_train[i_note][0][0:sample_limit]
            note = np.array(note, dtype=float)
            train_dataset.append(note)

train_dataset = np.array(train_dataset, dtype=float)


for i_note in range(ableton_valid.shape[0]):
    if i_note < notes_limit:
        note = ableton_valid[i_note][0][0:sample_limit]
        note = np.array(note, dtype=float)
        valid_dataset.append(note)

valid_dataset = np.array(valid_dataset, dtype=float)


for i_note in range(ableton_test.shape[0]):
    if i_note < notes_limit:
        note = ableton_test[i_note][0][0:sample_limit]
        note = np.array(note, dtype=float)
        test_dataset.append(note)

test_dataset = np.array(test_dataset, dtype=float)

log = Monitoring(experiment, TRAINING)

if TRAINING:

    network = object

    if experiment['number_of_layers'] == 1:
        from OneLayer import OneLayer
        network = OneLayer()

    if experiment['number_of_layers'] == 2:
        from TwoLayer import TwoLayer
        network = TwoLayer()

    if experiment['number_of_layers'] == 3:
        from ThreeLayer import ThreeLayer
        network = ThreeLayer()

    model = network.create_model(experiment, train_dataset, train_dataset, log, random=experiment['random'])

    # load model
    export_model.save_model(model, log)

else:

    id_experiment = 1
    load_model = export_model.load_model(id_experiment, log)

    x = np.arange(22050)
    signal_1 = np.array(np.sin(2 * np.pi * 550 * x / 22050), dtype=float)

    saw = np.array(signal.sawtooth(2 * np.pi * 550 * x / 22050))

    write(log.main_folder+log.samples_folder+log.samples_folder_basic+'basic_saw_550.wav', 22050, saw)
    y_saw = saw.reshape(1, 22050, 1)

    audio = GenerateAudio(fs=22050, seqs=1)
    audio.generate_sample(y_saw, load_model, log)

