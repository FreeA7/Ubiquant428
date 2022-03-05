# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:37:10 2022

@author: FreeA7
"""


import pandas as pd
import matplotlib.pyplot as plt


CSV = './original_data/split_data/2140.csv'


def adf_check(data):
    '''
    ADF检验
    ADF(Augmented Dickey-Fuller) 单位根检验方法
    H0：具有单位根，属于非平稳序列。
    H1：没有单位根，属于平稳序列，说明这个序列不具有时间依赖型结构。
    t值越小，越可以拒绝H0；p > 0.05，不能拒绝原假设

    Parameters
    ----------
    data : Dataframe
        时间序列

    Returns
    -------
    ADF_output : Dataframe
        ADF结果

    '''
    from statsmodels.tsa.stattools import adfuller
    
    
    ADF_result = adfuller(data)
    ADF_output = pd.Series(ADF_result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in ADF_result[4].items():
        ADF_output['Critical Value (%s)'%key] = '%.5f' % value
    
    return ADF_output


if __name__ == '__main__':
    data = pd.read_csv(CSV, header = None)
    data = data[[1, 3]]
    data.columns = ['time', 'target']
    data.sort_values(by = 'time', axis = 0, ascending = False, inplace = True)
    data.set_index('time')
    
    fig = plt.figure(figsize = (10, 20), dpi = 200)
    ax_time_series = fig.add_subplot(411)
    mean = data['target'].mean()
    ax_time_series.plot(data['time'], [mean for _ in range(data.shape[0])], 
                        color = 'red', linestyle = '--', linewidth = '1', 
                        label = 'mean:%.2f' % mean)
    ax_time_series.plot(data['time'], data['target'], color='blue', label = 'target')
    ax_time_series.set_xlabel('time')
    ax_time_series.set_ylabel('target')
    ax_time_series.set_title('time-target')
    ax_time_series.legend() 
    
    # 自相关性画图
    from statsmodels.graphics.tsaplots import plot_acf
    
    ax_autocorrelation = fig.add_subplot(412)
    plot_acf(data['target'], ax_autocorrelation, lags = 100)
    ax_autocorrelation.set_title('Autocorrelation of Target')
    
    print('target的ADF结果为：')
    print(adf_check(data['target']))
    
    

# =============================================================================
#     # =============================================================================
#     # 选取阶数
#     #     'aic_min_order': (4, 5),
#     #     'bic_min_order': (0, 1),
#     #     'hqic_min_order': (0, 1)
#     # =============================================================================
#     from statsmodels.tsa.stattools import arma_order_select_ic
#     
#     
#     order = arma_order_select_ic(data['target'], max_ar = 6, max_ma = 6,
#                                  ic = ['aic', 'bic', 'hqic'])
#     print(order.aic_min_order)
# =============================================================================

    
    
    from statsmodels.tsa.stattools import ARMA
    from statsmodels.graphics.api import qqplot
    from statsmodels.api import stats
    
    
    model_arma = ARMA(data['target'], (4, 2)).fit()
    resid = model_arma.resid
    ax_resid = fig.add_subplot(413)
    fig = qqplot(resid, line = '45', ax = ax_resid, fit = True)
    ax_resid.set_title('QQ of Resid')
    
    ax_resid_autocorrelation = fig.add_subplot(414)
    plot_acf(resid, ax_resid_autocorrelation, lags = 100)
    ax_resid_autocorrelation.set_title('Autocorrelation of Resid')
    
    print('ARMA(4, 2)的残差的DW检验量为%f' % stats.durbin_watson(resid))
    
    plt.show()