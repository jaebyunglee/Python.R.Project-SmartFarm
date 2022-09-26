import os
import re
import math
import copy
import warnings
import datetime
import numpy as np
import pandas as pd


class eval():
    def __init__(self, model_res):
        self.model_res = model_res
        self.y_name    = self.model_res['yvar_name']
        self.thrs      = np.where(self.y_name == 'y_암모니아',70,0.2)
        self.eval_result()
        
    def eval_result(self):
        
        '''
        - MAPE 계산 함수
        '''
        def _calc_mape(y_true, y_pred):
            mape_val = abs((y_true - y_pred)/(y_true + 1e-10))
            mape_val = np.where(mape_val > 1 ,1, mape_val)
            return np.nanmean(mape_val)
        
        '''
        - ACC 계산 함수
        '''
        def _calc_acc( y_true, y_pred, thrs):
            y_true = (y_true > thrs) + 0
            y_pred = (y_pred > thrs) + 0
            tp = np.nansum((y_true == 1) & (y_pred == 1))
            tn = np.nansum((y_true == 0) & (y_pred == 0))
            fp = np.nansum((y_true == 0) & (y_pred == 1))
            fn = np.nansum((y_true == 1) & (y_pred == 0))
            return (tp+tn)/(tp+tn+fp+fn)
        
        
        # 모델 평가 및 BEST 모델 선정
        eval_result = []
        mdl_list = [s for s in self.model_res.keys() if '_ret' in s]
        
        for mdl in mdl_list:
            model_name = self.model_res[mdl]['model_name']
            y_true     = self.model_res[mdl]['test_res'][self.y_name].values
            y_pred     = self.model_res[mdl]['test_res']['pred'].values
            mape       = _calc_mape(y_true, y_pred)
            acc        = _calc_acc (y_true, y_pred, self.thrs)
            print(f'> {model_name.ljust(4," ")} / MAPE : {mape :.4f} / ACC : {acc : .4f}')
            eval_result.append(pd.DataFrame({'y_name' : self.y_name, 'model' : model_name,"mape" : mape,"acc" : acc}, index = [0]))

        self.eval_res = pd.concat(eval_result).\
                        sort_values(['mape','acc'], ascending = [True,False]).\
                        reset_index(drop = True)   
        
        print(f'BEST Model is {self.eval_res.iloc[0]["model"].ljust(4," ")} / MAPE : {self.eval_res.iloc[0]["mape"] : .4f} / ACC : {self.eval_res.iloc[0]["acc"] : .4f}')
        return self.eval_res
    
    