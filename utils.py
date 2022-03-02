# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:55:24 2022

@author: FreeA7
"""



class Record:
    def __init__(self, time, data):
        self.time = time
        self.data = data


class Product:
    def __init__(self, id, target=None):
        self.id = id
        self.target = target
        self.records = {}
        
    def add_record(self, time, data):
        self.records[time] = Record(time, data)
        
        
    
        
            