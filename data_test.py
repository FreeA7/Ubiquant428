# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 23:51:04 2022

@author: FreeA7
"""

import logging
import os
import multiprocessing


logging.basicConfig(level=logging.INFO, filename = './logs/data_test.log', filemode = 'w',
                format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

NUM_WORKER = 15

# =============================================================================
# # 读取原始数据并将数据按照产品ID拆分成不同的csv
# count = 0
# with open('./original_data/train.csv', 'r') as f:
#     f.readline()
#     while 1:
#         line = f.readline()
#         if not line:
#             break
#         line_data = line[:-1].split(',')
#         investment_id = line_data[2]
#         with open('./original_data/split_data/%s.csv' % investment_id, 'a') as w:
#             w.write(line)
#         count += 1
#         if count % 10000 == 0:
#             logging.info('Have split %d rows' % count)
# =============================================================================


import pandas as pd
import numpy as np
from time import sleep


# 读取每一个csv并计算每一个alpha与target的相关系数
def get_corr(columns, csvs, input_queue, output_queue):
    while not input_queue.empty():
        i = input_queue.get()
        data_all = pd.DataFrame(columns = columns, dtype = np.float32)
        count = 0
        for file in csvs:
            count += 1
            data = pd.read_csv(file, header = None)
            data = data[[i*10 + j + 4 for j in range(10)] + [3]]
            data.columns = columns
            data_all = data_all.append(data)
            output_queue.put('Have read index %d - %s - %d get' % (i, file, count))
        io_con = ''
        for j in range(10):
            io_con += 'f_%d,%.6f\n' % (i*10 + j, data_all[['alpha%d' % j, 'target']].corr().iat[0, 1])
        output_queue.put((1, io_con))
    output_queue.put(1)
        

if __name__ == '__main__':
    csvs = []
    for root, dirs, files in os.walk('./original_data/split_data/'):
        for file in files:
            if file.endswith('.csv'):
                csvs.append(root + file)
                
    columns = ['alpha%d' % i for i in range(10)] + ['target']
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    output_queue = manager.Queue()
    for i in range(30):
        input_queue.put(i)
        
    pool = multiprocessing.Pool(NUM_WORKER)
    for i in range(NUM_WORKER):
        pool.apply_async(get_corr, args = (columns, csvs, input_queue, output_queue))
    pool.close()
        
    count = 0
    while 1:
        sleep(0.1)
        if not output_queue.empty():
            con = output_queue.get()
            if isinstance(con, str):
                print(con)
                logging.info(con)
            elif isinstance(con, int):
                count += 1
                if count == NUM_WORKER:
                    break
            elif isinstance(con, tuple):
                with open('./statistical_analysis/corr.csv', 'a') as w:
                    w.write(con[1])
            else:
                raise TypeError('ERROR')
  