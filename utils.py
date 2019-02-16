import os


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass