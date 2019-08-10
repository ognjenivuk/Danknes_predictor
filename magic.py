import matplotlib.pyplot as plt

_train, _valid = [], []
_trainplt, _valplt = None, None

def init_loss():
    global _train, _valid, _trainplt, _valplt
    _train = []
    _valid = []
    plt.ion()
    _trainplt = plt.plot([0])
    _valplt = plt.plot([0])

def add_vals(train, valid):
    _train.append(train)
    _valid.append(valid)
    _trainplt.append(train)
    _valplt.append(valid)
    plt.draw()