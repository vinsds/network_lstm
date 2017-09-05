import numpy as np
from scipy.io.wavfile import write
import os
import sys

class GenerateAudio:

    def __init__(self, seqs, fs):
        self.n_seqs = seqs
        self.fs = fs

    def generate_sample(self, input_signal, model, log):

        folder = log.main_folder + log.samples_folder

        for dirpath, dirnames, files in os.walk(folder):
            if str(files).find('.wav') != -1:
                print 'Folder contains wav files, exporting stopped'
                sys.exit()

        if not os.path.isdir(folder):
            os.mkdir(folder)

        seqs = self.n_seqs

        audio_export = []

        input_signal = np.array(input_signal, dtype=float)

        for t in range(seqs):
            audio_export = []

            print "Creating sample #", t
            # y = valid_dataset[t].reshape(1, valid_dataset[t].shape[0], 1)
            #TODO eliminare 22050, inserire parametro variabile
            wave = input_signal.reshape(1, 22050, 1)

            for i in range(1):

                predict = model.predict(wave)[0]  # 500, 256

                for index, samples in enumerate(predict):
                    samples = np.array(samples)
                    index_ar = samples.argmax()
                    audio_export.append(samples[index_ar])

            data = np.array(audio_export)
            data = self.mu2linear(data)
            data = data.astype('float32')
            data -= data.min()
            data /= data.max()
            data -= 0.5
            data *= 0.95

            write(folder+'/sample_no_'+str(t)+'.wav', self.fs, data)

        print "Samples saved into -->", folder

    @staticmethod
    def mu2linear(x, mu=255):
        mu = float(mu)
        x = x.astype('float32')
        y = 2. * (x - (mu + 1.) / 2.) / (mu + 1.)
        return np.sign(y) * (1. / mu) * ((1. + mu) ** np.abs(y) - 1.)


