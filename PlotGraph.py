import matplotlib.pyplot as plt
import numpy as np


class PlotGraph:
    def __init__(self, history_path, wav_path):
        self.history_path = history_path
        self.wav_path = wav_path

    def plot_loss(self):
        f = open(self.history_path, 'r')
        loss = []
        val_loss = []

        lines = str(f.readlines())

        if lines.find('val_loss') != -1:
            x = lines.replace("val_loss", " | val_loss").split('|')[1]
            x = x.replace('val_loss\':', '').replace('[', '').replace(']', '').replace('}', '').replace('\"','')
            split_val_values = x.split(',')
            for i in range(len(split_val_values)):
                val_loss.append(split_val_values[i])

        if lines.find('loss') !=- 1:
            x = lines.replace("val_loss", " | val_loss").split('|')[0]
            loss_values = x.replace("[\"{'loss': [", '').replace(']', ' ').replace('\'','').replace("}","").replace('\"','')
            split_loss_values = loss_values.split(',')
            for i in range(len(split_loss_values)):
                if not split_loss_values[i].isspace():
                    loss.append(split_loss_values[i])


        f.close()

        loss = np.array(loss, dtype=float)
        val_loss = np.array(val_loss, dtype=float)

        plt.plot(loss)
        plt.plot(val_loss)
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.show()

    def plot_wav(self):
        print "TODO"
        #TODO implementare wave plot
        #plt.plot(x, data)
        #plt.xlabel('amp')
        #plt.ylabel('time')
        #plt.show()

