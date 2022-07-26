# save models and data in each format.

import joblib
import yaml
import pickle
import os

def remove_file(filename):
    os.system('rm -rf {}'.format(filename))


def save_model(model, savefilename, overwrite=True):
    if overwrite:
        remove_file(savefilename)
    joblib.dump(model, savefilename, compress = 1)
    print(savefilename, 'has been saved.')


def load_model(savefilename):
    loaded_model = joblib.load(savefilename)
    return loaded_model


def save_data(data, savefilename, overwrite=True):
    if overwrite:
        remove_file(savefilename)
    pickle.dump(data, open(savefilename, 'wb'))
    print(savefilename, 'has been saved.')


def load_data(savefilename):
    data = pickle.load(open(savefilename, 'rb'))
    return data


def save_yaml(data, savefilename, overwrite=True):
    if overwrite:
        remove_file(savefilename)
    yaml.dump(data, open(savefilename, 'w'), sort_keys=False)
    print(savefilename, 'has been saved.')


def load_yaml(savefilename):
    data = yaml.load(open(savefilename, 'r'), Loader=yaml.FullLoader)
    return data
