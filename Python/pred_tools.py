import os
import re
import math
import copy
import pickle
import shutil
import warnings
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

'''
예측 클래스
'''
class pred():
    
    def __init__(self, working_date, y_name, model_path, data):
        self.working_date = working_date
        self.y_name       = y_name
        self.model_path   = model_path
        self.data         = data
        self._model_load_fn()
        self._pred_fn()

    def _model_load_fn(self):
        self.best_model_path = os.path.join(self.model_path,self.working_date,self.y_name,'BEST')
        self.best_model_name = os.listdir(self.best_model_path)[0].split('.')[0]
        
        print(f'Working Date : {self.working_date} / Y Name : {self.y_name} / BEST MODEL is {self.best_model_name}')
        with open(os.path.join(self.best_model_path,self.best_model_name + '_info.pkl'), 'rb') as f:
            self.model_info = pickle.load(f)

        if self.model_info['model_name'] in ['ANN','LSTM']:
            # 모델정보 불러오기
            self.model = tf.keras.models.load_model(self.model_info['model'])
        else :
            with open(self.model_info['model'], 'rb') as f: 
                self.model = pickle.load(f)
                
    def _pred_fn(self):
        dat        =  copy.deepcopy(self.data)
        model      = self.model
        model_info = self.model_info

        if   model_info['model_name'] == 'ANN'  :
            tmp_pred_dat = dat['pred'][model_info['scale_info']['xvar_name']]

            def _scaliling(xx):
                min_val = model_info['scale_info']['min']
                max_val = model_info['scale_info']['max']
                return (xx - min_val)/(max_val - min_val + 1e-10)

            tmp_pred_dat = np.apply_along_axis(_scaliling,1,tmp_pred_dat)
            tmp_pred_dat[tmp_pred_dat < 0] = 0
            tmp_pred_dat[tmp_pred_dat > 1] = 1
            dat['pred']['pred_val'] = model.predict(tmp_pred_dat)

        elif model_info['model_name'] == 'LSTM' :
            tmp_pred_dat = dat['pred'][model_info['scale_info']['xvar_name']]

            # LSTM 전처리 데이터 생성
            nTimeSteps = model_info['nTimeSteps']
            nInterval  = model_info['nInterval']

            # 디멘전 배치 사이즈
            dim_batch = tmp_pred_dat.shape[0] - (nInterval * (nTimeSteps - 1))

            # 데이터 인덱스 생성
            idx_list = []
            for i in range(dim_batch):
                idx_list.append(np.arange(start = i, stop = nInterval * (nTimeSteps - 1) + i + 1, step = nInterval))

            # LSTM 시계열 데이터 생성    
            x_array = []
            date_df = []

            for idx in range(len(idx_list)):
                x_array.append(np.array(tmp_pred_dat.iloc[idx_list[idx]]))
            x_array = np.array(x_array)

            # Min, Max Scaling
            min_val = model_info['scale_info']['min']
            max_val = model_info['scale_info']['max']

            def _scaliling(xx):
                return (xx - min_val)/(max_val - min_val + 1e-10)

            x_array = np.apply_along_axis(_scaliling, 2, x_array)
            x_array[x_array < 0] = 0
            x_array[x_array > 1] = 1

            # pred
            pred = model.predict(x_array)[:,-1,:]
            pred = np.append(np.repeat(np.nan,nTimeSteps-1),pred)
            dat['pred']['pred_val'] = pred

        else :
            tmp_pred_dat = dat['pred'][model_info['xvar']]
            if model_info['model_name'] == 'XGB':
                dat['pred']['pred_val'] = model.predict(xgb.DMatrix(tmp_pred_dat))
            else :
                dat['pred']['pred_val'] = model.predict(np.array(tmp_pred_dat))

        self.pred_ret = dat['pred']         