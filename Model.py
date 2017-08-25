from keras.models import model_from_json


class Model:

    def __init__(self):
        print " "

    def save_model(self, model, experiment):

        file_name = 'id_%s__neurons_%s__epochs_%s' % (
            experiment['id_experiment'],
            experiment['lstm_neurons'],
            experiment['n_epochs']
        )

        print "Model Export:"
        model_json = model.to_json()

        json_string = "./export_models/model_json/"+str(file_name)+".json"
        weight_string = "./export_models/weight_h5/"+str(file_name)+".h5"

        with open(json_string, "w") as json_file:
            json_file.write(model_json)
            model.save_weights(weight_string)
        print "Model saved "

    def load_model(self, path_json, path_weight):
        json_string = "./export_models/model_json/"+str(path_json)+".json"
        weight_string = "./export_models/weight_h5/"+str(path_weight)+".h5"
        json_file = open(json_string, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weight_string)

        print("Model Loaded")

        return loaded_model


