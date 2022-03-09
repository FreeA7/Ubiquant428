# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:34:19 2022

@author: FreeA7
"""


import abc


class Model:
    @abc.abstractmethod
    def fit(self, x, y):
        pass
    
    @abc.abstractmethod
    def predict(self, x):
        pass
        