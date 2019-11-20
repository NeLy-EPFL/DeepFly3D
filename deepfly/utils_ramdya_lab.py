import re
from logging import getLogger
import numpy as np

known_users = [  
    (r'/CLC/', [0, 6, 5, 4, 3, 2, 1]),
    (r'data/test', [0, 1, 2, 3, 4, 5, 6])
]


def find_default_camera_ordering(input_folder):
    input_folder = str(input_folder)  # allows input_folder to be a pathlib.Path instance
    candidates = [ordering for (regex, ordering) in known_users if re.search(regex, input_folder)]
    if candidates:
        getLogger('df3d').debug(f'Default camera ordering found for current user: {candidates[0]}')
        return np.array(candidates[0])
