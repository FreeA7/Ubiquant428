# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:46:00 2022

@author: FreeA7
"""


from sklearn.linear_model import LinearRegression
from utils.model import Model


class LeastSquare(Model):
    def __init__(self):
        super(LeastSquare, self).__init__()
        self.model = LinearRegression()
    
    def fit(self, x, y):            
        self.model.fit(x, y)
        
    def predict(self, x):
        return self.model.predict(x)
    
            