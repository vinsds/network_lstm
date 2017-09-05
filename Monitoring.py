import os
import sys


class Monitoring:

    main_folder = ''

    def __init__(self, experiment, training):
        self.root = '.experiments'
        # create folders
        self.main_folder = './experiments/experiment_id_'+str(experiment['id_experiment'])
        self.model_folder = '/model/'
        self.weight_folder = '/weight/'
        self.samples_folder = '/wav_samples/'
        self.samples_folder_basic = '/basic/'

        # create log file
        self.file_name = 'lstm_layers_%s__neurons_%s__epochs_%s' % (
            experiment['number_of_layers'],
            experiment['lstm_neurons'],
            experiment['n_epochs']
        )

        self.model_file = 'model_%s' % (
            experiment['id_experiment']
        )

        self.weight_file = 'weight_%s' % (
            experiment['id_experiment']
        )

        self.loss_file = 'loss_%s' % (
            experiment['id_experiment']
        )

        if training:
            if not os.path.isdir(self.main_folder):
                print 'Folder '+self.main_folder+' created'
                os.mkdir(self.main_folder)
                os.mkdir(self.main_folder + self.model_folder)
                os.mkdir(self.main_folder + self.weight_folder)
                os.mkdir(self.main_folder + self.samples_folder)
                os.mkdir(self.main_folder + self.samples_folder + self.samples_folder_basic)
                self.create_log_file(experiment, self.main_folder, self.file_name)
            else:
                print 'Folder exist, training stopped'
                sys.exit()

    @staticmethod
    def create_log_file(experiment, main_folder, file_name):
        print 'Creating log file...'
        logs = open(main_folder+'/'+file_name+'.txt', 'w')
        logs.write('Id experiment:\t'+str(experiment['id_experiment'])+"\n")
        logs.write('Number of layers:\t'+str(experiment['number_of_layers'])+"\n")
        logs.write('LSTM Neurons:\t'+str(experiment['lstm_neurons'])+"\n")
        logs.write('Number of epochs:\t'+str(experiment['n_epochs'])+"\n")
        logs.write('Q_levels:\t'+str(experiment['q_levels'])+"\n")
        logs.write('Activation LSTM Layer:\t'+str(experiment['lstm_activation'])+"\n")
        logs.write('Activation Dense Layer:\t'+str(experiment['dense_activation'])+"\n")
        logs.write('Batch Size:\t'+str(experiment['batch_size'])+"\n")
        logs.write('Random Samples:\t'+str(experiment['random'])+"\n")
        logs.write('Number of sample for training:\t'+str(experiment['train_set_shape'])+"\n")
        print 'Log file created'
        logs.close()