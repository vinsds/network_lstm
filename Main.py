from Logs import *
import numpy as np
from Model import *
from GenerateAudio import *

experiment = {
    'id_experiment': 2,
    'lstm_neurons': 100,
    'n_epochs': 5,
    'q_levels': 256,
    'lstm_activation': 'softmax',
    'dense_activation': 'relu',
    'batch_size': 2,
    'number_of_layers': 1,
    'random': False
}

log = Logs(experiment)

# import datasets
ableton_train = np.load('../music-Ableton_new/ableton_train.npy')
ableton_valid = np.load('../music-Ableton_new/ableton_valid.npy')
ableton_test = np.load('../music-Ableton_new/ableton_test.npy')

# cut the notes
train_dataset = []
valid_dataset = []

sample_limit = 22050
notes_limit = 9

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

model = network.create_model(experiment, train_dataset, valid_dataset, random=experiment['random'])

export_model = Model()
export_model.save_model(model, experiment)

load_model = export_model.load_model("id_2__neurons_100__epochs_5", "id_2__neurons_100__epochs_5")

audio = GenerateAudio(fs=22050, seqs=2)

audio.generate_sample(train_dataset, load_model, experiment)