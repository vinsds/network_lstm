import keras
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.optimizers import Adam


class TwoLayer:

    def __init__(self):
       print 'created...'

    def create_model(self, experiment, train_dataset, random=False):
        data_input = Input(
            batch_shape=(
                1,
                train_dataset.shape[1],
                1
            ),
            name='input'
        )

        _lstm = LSTM(
            experiment['lstm_neurons'],
            return_sequences=True,
            activation=experiment['lstm_activation']
        )(data_input)
        
        __lstm = LSTM(
            experiment['lstm_neurons'],
            return_sequences=True,
            activation=experiment['lstm_activation']
        )(_lstm)

        output = Dense(
            experiment['q_levels'],
            activation=experiment['dense_activation']
        )(__lstm)

        model = Model(
            inputs=[
                data_input
            ],
            outputs=output
        )

        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.compile(loss='categorical_crossentropy', optimizer=Adam())
        model.summary()

        self.training(model, experiment, train_dataset, random)
        return model

    @staticmethod
    def training(model, experiment, train_dataset, random=False):
        for item in range(train_dataset.shape[0]):
            if random:
                from random import randint
                s = train_dataset[randint(0, train_dataset.shape[0])]
            else:
                s = train_dataset[item]

            x_train = s.reshape(1, s.shape[0], 1)
            y_train = keras.utils.to_categorical(x_train, num_classes=256)
            y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1]))

            #x_valid = valid_dataset[item].reshape(1, valid_dataset[item].shape[0], 1)
            #y_valid = keras.utils.to_categorical(x_valid, num_classes=256)
            #y_valid = y_valid.reshape((1, y_valid.shape[0], y_valid.shape[1]))

            hist = model.fit(
                x_train,
                y_train,
                #validation_data=(
                #    x_valid,
                #    y_valid
                #),
                epochs=experiment['n_epochs'],
                batch_size=experiment['batch_size']
            )
            return hist