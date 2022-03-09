# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:32:48 2022

@author: FreeA7
"""


import os

import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from least_squares import LeastSquare


csvs = []
for root, dirs, files in os.walk('./original_data/split_data/'):
    for f in files:
        if f.endswith('.csv'):
            csvs.append(root + f)


CSV_RANGE = csvs[:100]
INDEX = 'row_id'
COLUMNS_NAME = ['row_id', 'time', 'investment_id', 'target'] + ['alpha%d' % i for i in range(300)]



if __name__ == '__main__':
    data = pd.DataFrame(columns = COLUMNS_NAME)
    for csv in CSV_RANGE:
        data = data.append(pd.read_csv(csv, header = None, names = COLUMNS_NAME))

    
    x_data = data[['row_id', 'investment_id', 'time'] + ['alpha%d' % i for i in range(300)]]
    y_data = data['target']
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3, random_state = 19950916)
    
    x_train = x_train.sort_index()
    x_test = x_test.sort_index()
    y_train = y_train.sort_index()
    y_test = y_test.sort_index()
    
    print('Origin data alphas : ', x_data.shape)
    print('Train data alphas : ', x_train.shape)
    print('Test data alphas : ', x_test.shape)
     
    print('Origin data target : ', y_data.shape)
    print('Train data target : ', y_train.shape)
    print('Test data target : ', y_test.shape)
    
    del data, x_data, y_data
    
    least_square_model = LeastSquare()
    least_square_model.fit(x_train.iloc[:,2:], y_train)
    
    test_predict = least_square_model.predict(x_test.iloc[:,2:])
    test_predict = pd.DataFrame(test_predict, columns = ['predict'])

    index = x_test[INDEX]
    index.reset_index(drop = True, inplace = True)
    
    y_test.reset_index(drop = True, inplace = True)
    
    result = pd.concat([index, test_predict, y_test], axis = 1)
    
    print('*********** least_square_model *********** ')
    print('RMSE : ', mean_squared_error(result['target'], result['predict']))
    # print('RMSLE : ', mean_squared_log_error(result['target'], result['predict']))
    print('R^2 : ', least_square_model.model.score(x_test.iloc[:,2:], y_test))
    
    # =============================================================================
    # *********** least_square_model *********** 
    # RMSE :  0.8372687556313455
    # R^2 :  0.012662855502432735
    # =============================================================================
    
    
    