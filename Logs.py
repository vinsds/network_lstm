import os
import sys

class Logs:

    def __init__(self, experiment):
        #folder = './experiment_id_'+str(experiment['id_experiment'])

        #if not os.path.isdir(folder):
        #    os.mkdir(folder)
        #else:
        #    print "esiste"
        #print self.folder

        file_name = 'id_%s__neurons_%s__epochs_%s' % (
            experiment['id_experiment'],
            experiment['lstm_neurons'],
            experiment['n_epochs']
        )

        self.create_log_file(experiment, file_name)

    def create_log_file(self, experiment, file_name):
        print 'Writing log file...'
        logs = open('./logs_experiment/'+file_name+'.txt', 'w')
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

