""" Utility functions related to directories. """
import os


def maybe_make_directory(path):
    """ Makes a directory if doesn't exist.

    :param path: Path to directory to maybe make.
    :return: T/F if the directory already existed.
    """
    if os.path.isdir(path):
        return True
    os.makedirs(path)
    return False
