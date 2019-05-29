import numpy as np
class Glovar:
    paras=None

def set_para():
    Glovar.paras=np.load('vgg19.npy', encoding='latin1').item()

def get_para():
    return Glovar.paras
