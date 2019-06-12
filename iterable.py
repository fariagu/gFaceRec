from __future__ import absolute_import, division, print_function

class Iterable:
    def __init__(self, prediction, filename, label):
        self.prediction = prediction
        self.filename = filename
        self.label = label
