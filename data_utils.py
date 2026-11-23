import numpy as np
import scipy.io as sio

def load_data(filename):
    data = sio.loadmat(filename)
    Yt, Ct = data["Yt"], data["Ct"]
    Yv, Cv = data["Yv"], data["Cv"]
    return Yt, Ct, Yv, Cv