# -*- coding: utf-8 -*-


import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

currentPath = os.path.dirname(os.path.realpath(__file__))

# Add lib to PYTHONPATH
libPath = os.path.join(currentPath, 'lib')
add_path(libPath)

modelPath = os.path.join(currentPath, 'models')
add_path(modelPath)