import os


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass