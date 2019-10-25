from __future__ import absolute_import

import errno
import os


def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def isfile(fname):
    return os.path.isfile(fname)


def isdir(dirname):
    return os.path.isdir(dirname)


def join(path, *paths):
    return os.path.join(path, *paths)


from pathlib import Path
def find_leaf_recursive(path):
    print('Recursive scan...')
    l = list()
    for root, dirs, files in os.walk(path):
        if Path(root).name == 'images':
            print('Found working directory: {}'.format(root))
            l.append(root)
    return l
