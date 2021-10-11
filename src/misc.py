import pickle
import numpy as np

def save_pckl_data(name, var):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(var, f)

def load_pckl_data(name):
    if name[-4:] == ".pkl":
        with open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

def load_vector_from_file(file_name, normalise=True):
    L = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        for l in lines:
            l = l[1:-2].split(',')
            l = np.array(list(map(float, l)))
            if normalise:
                l = l / np.linalg.norm(l)
            L.append(l)
    return L