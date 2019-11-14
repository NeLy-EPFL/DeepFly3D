from pathlib import Path
import os
from collections import deque


def find_subfolders(path, name):
    """
    Implements a Breadth First Search algorithm to find all subfolders named `name`.
    Using a BFS allows to stop as soon as we find the target subfolder, without listing its content.
    Which is a performance improvement when target subfolders contain hundreds on thousands of images.
    """
    found = []
    openlist = deque()
    visited = set()
    
    openlist.append(Path(path))
    while openlist:
        current = openlist.popleft()
        if current.is_dir() and current not in visited:
            visited.add(current)
            if current.name == name:
                found.append(str(current))
            else:
                for child in current.iterdir():
                    openlist.append(child)
    return found