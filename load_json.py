import json
import os
import pickle

import argparse

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, Dropout
from keras.layers import Input   

from keras.models import Model


def load_dense(json_obj, inp_layer):
    units = json_obj['units']
    activation = json_obj.get('activation', None)
    return Dense(units, activation=activation)(inp_layer)


def initialize(input_shape):
    return Input(shape=input_shape)

def test_sample(model, X_data, y_data):
    rng = range(5,7)
    sample = model.predict(X_data[rng])
    print("Range:")
    print(rng)
    print("Labels:")
    print(y_data[rng])
    print("Predictions:")
    print(sample)


def load_pickle(json_network):
    data = pickle.load(open(json_network['dataset'], 'rb'))
    X_data = data['data']
    y_data = data['label']
    return X_data, y_data


def load_layers(json_network, inp_layer):
    layers = list()

    for layer in json_network['layers']:
        layer_in = layer['input']
        if layer_in == 0:
            layers.append(load_dense(layer, inp_layer))
            continue
        layers.append(load_dense(layer, layers[layer_in-1]))
    return layers


def parse_arguments():
    parser = argparse.ArgumentParser(description="JSON arguments")
    parser.add_argument('json', type=str, default=None,
                    help='The JSON file string')
    parser.add_argument('-f', dest='file', 
                        type=str, default=None,
                        help='Optional Location ')

    args = parser.parse_args()
    if args.json == None and args.file == None:
        raise IOError('JSON must be specified, by either a file or a string')
    return args


def load_json(args):
    if args.json is not None:
        return json.loads(args.json)
    else:
        return json.load(json.load(open(args.file, 'r')))


if __name__ == "__main__":

    args = parse_arguments()
    json_network = load_json(args)
    inp_layer = initialize(json_network['input_shape'])
    network_layers = load_layers(json_network, inp_layer) 

    X_data, y_data = load_pickle(json_network)

    model = Model(inp_layer, network_layers[-1])
    model.compile(json_network['optimizer'], json_network['loss'], metrics=['accuracy'])
    model.fit(X_data, y_data, batch_size=json_network['batch_size'], 
              epochs=json_network['epochs'], validation_split=json_network['split'],
              shuffle=True)

    test_sample(model, X_data, y_data)


