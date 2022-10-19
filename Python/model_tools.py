# 기본 패키지
import os
import re
import math
import copy
import random
import shutil
import datetime
import numpy as np
import pandas as pd

# 모델링 관련 패키지
import xgboost    as xgb
import lightgbm   as lgb
import pmdarima   as pmd
import tensorflow as tf

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold, train_test_split

from sklearn   import datasets
from sklearn   import linear_model
from sklearn   import metrics
from sklearn   import model_selection
from itertools import repeat

import keras_tuner
from   keras_tuner        import HyperModel, Hyperband
from   keras_tuner.tuners import SklearnTuner

# 랜덤 시드 설정
random.seed(1234)
tf.random.set_seed(1234)


'''
SklearnTuner 클래스
 - 각 Trial에서 print 끄기
'''
class CustomSklearnTuner(SklearnTuner):
    def search(self, *args, **kwargs):
        self._display.verbose = False
        return super().search(*args, **kwargs) #부모 클래스의 search

'''
모델링 클래스
'''
class model():
    def __init__(self, yvar_name : str, data : dict):
        self.yvar_name    = yvar_name # 종속변수 명
        self.data         = data      # 전처리 클래스의 최종 데이터 dict
        self.modeling_mlr()
        self.modeling_rf()
        self.modeling_arima()
        self.modeling_xgb()
        self.modeling_lgb()
        self.modeling_ann()
        self.modeling_lstm()
        
        
    def modeling_arima(self) :
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is ARIMA'.center(40))
        print('='*40)
        
        dat = copy.deepcopy(self.data)
        
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name

        # 학습, 검증 데이터 준비
        arima_train_y = np.array(dat['train_dat'][yvar_name])
        arima_train_x = np.array(dat['train_dat'][xvar_name])
        arima_test_x  = np.array(dat['test_dat'][xvar_name])
        
        # Arima Fitting
        arima_model = pmd.auto_arima(y=arima_train_y, X=arima_train_x)
        
        # 학습/검증데이터 예측
        dat['train_dat']['pred'] = arima_model.predict(X=arima_train_x, n_periods=arima_train_x.shape[0])
        dat['test_dat']['pred'] = arima_model.predict(X=arima_test_x, n_periods=arima_test_x.shape[0])
        
        self.arima_ret = dict({'model' : arima_model,'model_name' : 'ARIMA' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})
        return self.arima_ret
    
    def modeling_mlr(self) :
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is MLR'.center(40))
        print('='*40)
        
        dat = copy.deepcopy(self.data)
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name

        # 학습, 검증 데이터 준비
        mlr_train_y = np.array(dat['train_dat'][yvar_name])
        mlr_train_x = np.array(dat['train_dat'][xvar_name])
        mlr_test_x  = np.array(dat['test_dat'][xvar_name])

        # 회귀분석 Fitting
        mlr_model = LinearRegression()
        mlr_model.fit(X=mlr_train_x, y = mlr_train_y)

        # 학습/검증데이터 예측
        dat['train_dat']['pred'] = mlr_model.predict(X=mlr_train_x)
        dat['test_dat']['pred'] = mlr_model.predict(X=mlr_test_x)
        self.mlr_ret = dict({'model' : mlr_model,'model_name' : 'MLR' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})
        
        return self.mlr_ret    
    
    
    def modeling_rf(self) :
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is Random Forest'.center(40))
        print('='*40)
        dat = copy.deepcopy(self.data)
        
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name

        # 학습, 검증 데이터 준비
        rf_train_y = np.array(dat['train_dat'][yvar_name])
        rf_train_x = np.array(dat['train_dat'][xvar_name])
        rf_test_x  = np.array(dat['test_dat'][xvar_name])
        
        # Hyper Model Build 함수
        def _build_rf_model(hp):
            rf_model = RandomForestRegressor(n_estimators=hp.Int('n_estimators', 10, 50, step=10),
                                             max_depth=hp.Int('max_depth', 3, 10, step = 1), random_state = 1234)
            return rf_model
        
        # Keras Tuner
        tuner = CustomSklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=10,
                    num_initial_points=None,
                    seed = 1234),
                hypermodel=_build_rf_model,
                scoring=metrics.make_scorer(metrics.mean_squared_error),
                cv=model_selection.KFold(5), # KFold 외에 Train/Test Split 사용 시 -> model_selection.PredefinedSplit (Train Set을 -1로, Test Set을 1로 index list 생성)
                directory='.',
                project_name="hp_temp_folder",
                overwrite = True,
                logger = False)
        
        # Search - Hyper Parameters
        tuner.search(rf_train_x,rf_train_y)
        
        # Best Hyper Parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.values)
        
        # Final Model Build
        rf_model = tuner.hypermodel.build(best_hps)

        # Fit
        rf_model.fit(rf_train_x,rf_train_y)

        # 학습/검증데이터 예측
        dat['train_dat']['pred'] = rf_model.predict(rf_train_x)
        dat['test_dat']['pred']  = rf_model.predict(rf_test_x)
        self.rf_ret = dict({'model' : rf_model,'model_name' : 'RF' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})
        
        return self.rf_ret
    
    
    def modeling_xgb(self):
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is XGBoost'.center(40))
        print('='*40)
        dat = copy.deepcopy(self.data)
        
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name

        # XGBoost 학습 데이터 준비
        # 학습, 검증 데이터 준비
        xgb_train_y = np.array(dat['train_dat'][yvar_name])
        xgb_train_x = np.array(dat['train_dat'][xvar_name])
        xgb_test_x  = np.array(dat['test_dat'][xvar_name])
        
        
        # Hyper Model Build 함수
        def _build_xgb_model(hp):
            xgb_model = xgb.XGBRegressor(n_estimators = hp.Int('n_estimators', 10, 300, step=10),
                                         max_depth    = hp.Int('max_depth', 3, 10, step = 1),
                                         min_child_weight = hp.Float('min_child_weight', 1, 3, step = 0.2),
                                         random_state = 1234)
            return xgb_model
        
        
        # Keras Tuner
        tuner = CustomSklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=10,
                    num_initial_points=None,
                    seed = 1234),
                hypermodel=_build_xgb_model,
                scoring=metrics.make_scorer(metrics.mean_squared_error),
                cv=model_selection.KFold(5), # KFold 외에 Train/Test Split 사용 시 -> model_selection.PredefinedSplit (Train Set을 -1로, Test Set을 1로 index list 생성)
                directory='.',
                project_name="hp_temp_folder",
                overwrite = True,
                logger = False)
        
        # Search - Hyper Parameters
        tuner.search(xgb_train_x,xgb_train_y)
        
        # Best Hyper Parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.values)
        
        # Final Model Build
        xgb_model = tuner.hypermodel.build(best_hps)
        
        # Fit
        xgb_model.fit(xgb_train_x,xgb_train_y)
        

        # 학습/검증데이터 예측
        dat['train_dat']['pred'] = xgb_model.predict(xgb_train_x)
        dat['test_dat']['pred']  = xgb_model.predict(xgb_test_x)

        self.xgb_ret = dict({'model' : xgb_model,'model_name' : 'XGB' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})

        return self.xgb_ret
    
#     def modeling_xgb(self):
#         '''
#         * 입력
#         yvar_name : 종속변수 명
#         train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

#         * 출력
#         dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
#         '''
#         print(f'Model Type is XGBoost')
#         dat = copy.deepcopy(self.data)
#         # 학습에 사용할 설명변수 명 지정
#         xvar_name = dat['x_var']
#         yvar_name = self.yvar_name

#         # XGBoost 학습 데이터 준비
#         train_d_mat = xgb.DMatrix(data = dat['train_dat'][xvar_name], label = dat['train_dat'][yvar_name])

#         # Grid Search
#         params = {'max_depth':[2,5,7],
#                   'min_child_weight':[1.0,3.0],
#                   'colsample_bytree':[0.5,0.75]}
#         params_grid = pd.DataFrame(ParameterGrid(params))

#         score_list           = []
#         num_boost_round_list = []
#         for params_idx, params in params_grid.iterrows() :
#             params_tmp  = {'max_depth'       : int(params['max_depth']),
#                            'min_child_weight': float(params['min_child_weight']),
#                            'colsample_bytree': float(params['colsample_bytree'])}
#             xgb_cv      = xgb.cv(dtrain = train_d_mat, params = params_tmp, num_boost_round = 200, nfold = 3, 
#                                  early_stopping_rounds = 10, maximize = 0, verbose_eval= 0, seed =1234)
#             num_boost_round_list.append(xgb_cv.shape[0])
#             score_list.append(xgb_cv['test-rmse-mean'].iloc[-1])

#         # Find Best Parameter
#         params_grid['num_boost_round'] = num_boost_round_list
#         params_grid['score']           = score_list
#         best_params = params_grid.iloc[np.argmin(params_grid['score']),:]
#         xgb_train_params = {'max_depth'       : int(best_params['max_depth']),
#                             'min_child_weight': float(best_params['min_child_weight']),
#                             'colsample_bytree': float(best_params['colsample_bytree'])}
#         num_boost_round = int(best_params['num_boost_round'])    
#         print(f'== XGBoost Best Tuning Parameters ==')
#         print(f'num_boost_round : {num_boost_round}')
#         print(xgb_train_params)
        
#         # XGBoost Fitting
#         xgb_model = xgb.train(dtrain = train_d_mat, params = xgb_train_params, num_boost_round = num_boost_round)

#         # 학습/검증데이터 예측
#         dat['train_dat']['pred'] = xgb_model.predict(xgb.DMatrix(dat['train_dat'][dat['x_var']]))
#         dat['test_dat']['pred']  = xgb_model.predict(xgb.DMatrix(dat['test_dat'][dat['x_var']]))

#         self.xgb_ret = dict({'model' : xgb_model,'model_name' : 'XGB' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})

#         return self.xgb_ret
    
    def modeling_lgb(self):
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is LightGBM'.center(40))
        print('='*40)
        dat = copy.deepcopy(self.data)
        
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name

        # XGBoost 학습 데이터 준비
        # 학습, 검증 데이터 준비
        lgb_train_y = np.array(dat['train_dat'][yvar_name])
        lgb_train_x = np.array(dat['train_dat'][xvar_name])
        lgb_test_x  = np.array(dat['test_dat'][xvar_name])
        
        
        # Hyper Model Build 함수
        def _build_lgb_model(hp):
            lgb_model = lgb.LGBMRegressor(n_estimators = hp.Int('n_estimators', 10, 300, step=10),
                                          max_depth    = hp.Int('max_depth', 3, 10, step = 1),
                                          num_leaves   = hp.Int('num_leaves', 3, 64, step = 1),
                                          random_state = 1234)
            return lgb_model
        
        
        # Keras Tuner
        tuner = CustomSklearnTuner(
                oracle=keras_tuner.oracles.BayesianOptimizationOracle(
                    objective=keras_tuner.Objective('score', 'min'),
                    max_trials=10,
                    num_initial_points=None,
                    seed = 1234),
                hypermodel=_build_lgb_model,
                scoring=metrics.make_scorer(metrics.mean_squared_error),
                cv=model_selection.KFold(5), # KFold 외에 Train/Test Split 사용 시 -> model_selection.PredefinedSplit (Train Set을 -1로, Test Set을 1로 index list 생성)
                directory='.',
                project_name="hp_temp_folder",
                overwrite = True,
                logger = False)
        
        # Search - Hyper Parameters
        tuner.search(lgb_train_x,lgb_train_y)
        
        # Best Hyper Parameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(best_hps.values)
        
        # Final Model Build
        lgb_model = tuner.hypermodel.build(best_hps)
        
        # Fit
        lgb_model.fit(lgb_train_x,lgb_train_y)
        

        # 학습/검증데이터 예측
        dat['train_dat']['pred'] = lgb_model.predict(lgb_train_x)
        dat['test_dat']['pred']  = lgb_model.predict(lgb_test_x)

        self.lgb_ret = dict({'model' : lgb_model,'model_name' : 'LGB' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})

        return self.lgb_ret
    
    
#     def modeling_lgb(self) -> dict :
#         '''
#         * 입력
#         yvar_name : 종속변수 명
#         train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

#         * 출력
#         dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
#         '''
#         print(f'Model Type is LightGBM')
#         dat = copy.deepcopy(self.data)
#         # 학습에 사용할 설명변수 명 지정
#         xvar_name = dat['x_var']
#         yvar_name = self.yvar_name

#         # LightGBM 학습 데이터 준비
#         train_d_mat = lgb.Dataset(data = dat['train_dat'][xvar_name], label = dat['train_dat'][yvar_name])

#         # Grid Search
#         params = {'num_leaves' : [3,31],
#                   'learning_rate' : [0.1],
#                   'feature_fraction' : [1],
#                   'bagging_fraction' : [1],
#                   'max_bin' : [255]}
#         params_grid = pd.DataFrame(ParameterGrid(params))
#         score_list           = []
#         num_boost_round_list = []
#         for params_idx, params in params_grid.iterrows():
#             params_tmp = {'objective'        : 'regression',
#                           'boosting'         : "gbdt",
#                           'metric'           : 'rmse',
#                           'verbose'          : -1,
#                           'num_leaves'       : int(params['num_leaves']),
#                           'learning_rate'    : float(params['learning_rate']),
#                           'feature_fraction' : float(params['feature_fraction']),
#                           'bagging_fraction' : float(params['bagging_fraction']),
#                           'max_bin'          : int(params['max_bin'])}
#             lgb_cv     = lgb.cv(params = params_tmp, train_set = train_d_mat, num_boost_round = 200, nfold = 3, early_stopping_rounds = 10, 
#                                 verbose_eval= False, seed =1234, stratified=False)
#             num_boost_round_list.append(len(lgb_cv['rmse-mean']))
#             score_list.append(lgb_cv['rmse-mean'][-1])

#         # Find Best Parameter
#         params_grid['num_boost_round'] = num_boost_round_list
#         params_grid['score']           = score_list
#         best_params = params_grid.iloc[np.argmin(params_grid['score']),:]
#         lgb_train_params = {'objective'        : 'regression',
#                             'boosting'         : "gbdt",
#                             'metric'           : 'rmse',
#                             'verbose'          : -1,
#                             'num_leaves'       : int(best_params['num_leaves']),
#                             'learning_rate'    : float(best_params['learning_rate']),
#                             'feature_fraction' : float(best_params['feature_fraction']),
#                             'bagging_fraction' : float(best_params['bagging_fraction']),
#                             'max_bin'          : int(best_params['max_bin'])}
#         num_boost_round = int(best_params['num_boost_round'])    
#         print(f'== LightGBM Best Tuning Parameters ==')
#         print(f'num_boost_round : {num_boost_round}')
#         print(lgb_train_params)
        
#         # LightGBM Fitting
#         lgb_model     = lgb.train(params = params_tmp, train_set = train_d_mat, num_boost_round = num_boost_round)

#         # 학습/검증데이터 예측
#         dat['train_dat']['pred'] = lgb_model.predict(dat['train_dat'][dat['x_var']])
#         dat['test_dat']['pred']  = lgb_model.predict(dat['test_dat'][dat['x_var']])

#         self.lgb_ret = dict({'model' : lgb_model,'model_name' : 'LGB' , "yvar" : yvar_name, "xvar" : xvar_name, "train_res" : dat['train_dat'], "test_res" : dat['test_dat']})

#         return self.lgb_ret
    
    def modeling_ann(self):
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is ANN'.center(40))
        print('='*40)
        dat = copy.deepcopy(self.data)
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name
        ann = MODELING_ANN(yvar_name = yvar_name, data = dat, TuneMethod = 'BO') # BO : Bayesian Optimization , HB : Hyper Band
        self.ann_ret = ann.ret

    def modeling_lstm(self):
        '''
        * 입력
        yvar_name : 종속변수 명
        train_anal_dat : TRAIN_ANAL_DAT의 출력 딕셔너리

        * 출력
        dict           : 모델 및 모델 관련 정보들을 가지고 있는 딕셔너리
        '''
        print('='*40)
        print('Model Type is LSTM'.center(40))
        print('='*40)
        dat = copy.deepcopy(self.data)
        # 학습에 사용할 설명변수 명 지정
        xvar_name = dat['x_var']
        yvar_name = self.yvar_name
        lstm = MODELING_LSTM(yvar_name = yvar_name, data = dat, TuneMethod = 'BO') # BO : Bayesian Optimization , HB : Hyper Band
        self.lstm_ret = lstm.ret   
        
        
        
'''
ANN Model Class
'''
class MODELING_ANN():
    def __init__(self, yvar_name : str, data : dict, TuneMethod : str) :
        
        self.yvar_name  = yvar_name
        self.dat        = copy.deepcopy(data)
        self.xvar_name  = self.dat['x_var']
        self.TuneMethod = TuneMethod
        self._preprocess()
        self._fit()
        self._pred()
        self.ret = dict({'model' : self.model,'model_name' : 'ANN' , "yvar" : self.yvar_name, "xvar" : self.xvar_name, 
                         "train_res" : self.dat['train_dat'], "test_res" : self.dat['test_dat'], "scale_info" : self.scale_info})

    def _preprocess(self):
        
        # 전처리
        # Train Valid Split Point
        self.split_point = math.ceil(self.dat['train_dat'].shape[0] * 0.7)
        
        # Train
        self.train_x_array = np.array(self.dat['train_dat'][self.xvar_name])[0:self.split_point,]
        self.train_y_array = np.array(self.dat['train_dat'][self.yvar_name])[0:self.split_point]
        
        # Valid
        self.valid_x_array = np.array(self.dat['train_dat'][self.xvar_name])[self.split_point:,]
        self.valid_y_array = np.array(self.dat['train_dat'][self.yvar_name])[self.split_point:]
        
        # Test
        self.test_x_array = np.array(self.dat['test_dat'][self.xvar_name])
        self.test_y_array = np.array(self.dat['test_dat'][self.yvar_name])

        # Min-Max Scaling
        self.scale_info = pd.DataFrame({'xvar_name' : self.xvar_name, 
                                        'min' : np.apply_along_axis(min, 0, self.train_x_array), 
                                        'max' : np.apply_along_axis(max, 0, self.train_x_array)})
        def _scaliling(xx):
            return (xx - self.scale_info['min'])/(self.scale_info['max'] - self.scale_info['min'] + 1e-10)
        self.x_tr_scale = np.apply_along_axis(_scaliling ,1,self.train_x_array)
        self.x_vd_scale = np.apply_along_axis(_scaliling ,1,self.valid_x_array)
        self.x_vd_scale[self.x_vd_scale < 0] = 0
        self.x_vd_scale[self.x_vd_scale > 1] = 1
        self.x_te_scale = np.apply_along_axis(_scaliling ,1,self.test_x_array)
        self.x_te_scale[self.x_te_scale < 0] = 0
        self.x_te_scale[self.x_te_scale > 1] = 1

#     def _fit(self):
#         # ANN Build
#         input_shape = self.train_x_array.shape[1] # input shape 설정
#         h_units     = [16]                  # 모델 Hidden Units 설정
#         self.ann = ann_model(input_shape,h_units) # 모델 Build
        
#         # Early Stopping, Reduce Learning Rate, HIstory
#         EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5, restore_best_weights=True, verbose = 0) 
#         reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', factor=0.5, patience=5, verbose=0, min_lr=1e-5)
#         history = tf.keras.callbacks.History()

#         self.ann.model.fit(self.x_tr_scale, self.train_y_array, validation_split = 0.3
#                                                , epochs = 100
#                                                , batch_size = 16
#                                                , callbacks = [EarlyStopping, reduce_lr, history]
#                                                , verbose = 0)

    def _fit(self):
        # ANN Build
        input_shape = self.train_x_array.shape[1] # input shape 설정
        
        if self.TuneMethod == 'BO' :
            print('='*40)
            print('Tuning Method : Bayesian Optimization'.center(40))    
            print('='*40)
            # BasianOptimization Tuning 
            tuner = keras_tuner.BayesianOptimization(ann_model(input_shape), 
                                                 objective = 'val_loss', 
                                                 max_trials=10, # 튜닝 파라미터 시도 회수
                                                 num_initial_points=2,
                                                 executions_per_trial=2, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        if self.TuneMethod == 'HB' :
            print('='*40)
            print('Tuning Method : Hyper Band'.center(40))
            print('='*40)
            # HyperBand Tuning    
            tuner = keras_tuner.Hyperband(ann_model(input_shape), 
                                                 objective = 'val_loss', 
                                                 max_epochs=5, 
                                                 factor = 3,
                                                 executions_per_trial=2, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        TryNum = 10    
        for i in range(TryNum):
            try :
                print(f'Tuner Search Try - {str(i).zfill(2)}')
                tuner.search(self.x_tr_scale, 
                             self.train_y_array, 
                             epochs = 10,
                             validation_data = (self.x_vd_scale, self.valid_y_array), # validation data를 입력해 줄 수 있음 (fit 함수와 동일)
                             verbose = 0,
                             batch_size = 16)
            except :
                # 에러 발생 시 다음 Try 시도
                continue
           
            # 에러 발생 하지 않을 시 루프 종료
            break

        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print(best_hps.values)
        self.model = tuner.hypermodel.build(best_hps)
        self.model.summary()

        # Early Stopping 
        """
        - restore_best_weights -
        Whether to restore model weights from the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of training are used
        """
        self.EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10,
                                                         restore_best_weights=True, verbose = 1) 

        # reduce_lr
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='min', min_lr=1e-5)


        # # 모델 Check Point 저장
        # ModelCheck = tf.keras.callbacks.ModelCheckpoint(filepath = 'D:/workspace/OUT/best_model_gru/best_model_gru_{}.h5'.format(self.model_num)
        #                                                 , monitor = 'val_loss'
        #                                                 , mode = 'min'
        #                                                 , save_best_only = True)

        # History
        self.history = tf.keras.callbacks.History()

        # 에폭 10까지 학습
        self.model.fit(self.x_tr_scale, self.train_y_array, epochs = 10, validation_data = (self.x_vd_scale, self.valid_y_array)
                       , batch_size = 16
                       , callbacks = [self.history]
                       , verbose=0)


        # 에폭 10부터 학습
        self.model.fit(self.x_tr_scale, self.train_y_array
                      , validation_data = (self.x_vd_scale, self.valid_y_array)
                      , initial_epoch = 10
                      , epochs = 100
                      , batch_size = 16
                      , callbacks = [self.EarlyStopping, self.reduce_lr, self.history]
                      , verbose=0)
        
        self.train_loss = self.model.history.history['loss']
        self.valid_loss = self.model.history.history['val_loss']
        self.epochs = self.model.history.epoch


    def _pred(self):
        self.dat['train_dat']['pred'] = self.model.predict(np.vstack([self.x_tr_scale,self.x_vd_scale]), verbose = 0)
        self.dat['test_dat']['pred'] = self.model.predict(self.x_te_scale, verbose = 0)
            
## ANN Model Class
class ann_model(HyperModel):
    def __init__(self, input_shape : int):
        self.input_shape = input_shape

    def build(self, hp):
        input_layer  = tf.keras.Input(shape = self.input_shape, name = 'input_layer')
        for x in range(hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)):
            if x == 0:
                ann_layer = tf.keras.layers.Dense(hp.Int('units_{}'.format(str(x+1)), min_value = 4, max_value = 32, step = 4),
                                                  activation = 'relu', name = f'ann_layer_{str(x+1)}')(input_layer)
            else :
                ann_layer = tf.keras.layers.Dense(hp.Int('units_{}'.format(str(x+1)), min_value = 4, max_value = 32, step = 4), 
                                                  activation = 'relu', name = f'ann_layer_{str(x+1)}')(ann_layer)
        output_layer   = tf.keras.layers.Dense(1, activation = 'linear', kernel_initializer = tf.keras.initializers.orthogonal(seed = 1234),  name = 'output_layer')(ann_layer)
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        
        # compile 
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'mse')
        return model
        

        
        
'''
LSTM Model Class
'''
class MODELING_LSTM():
    def __init__(self, yvar_name : str, data : dict, TuneMethod : str) :
        self.yvar_name = yvar_name
        self.dat       = copy.deepcopy(data)
        self.xvar_name = self.dat['x_var']
        self.TuneMethod = TuneMethod
        self._preprocess()
        self._fit()
        self._pred()
        self.ret = dict({'model' : self.model,'model_name' : 'LSTM' , "yvar" : self.yvar_name, "xvar" : self.xvar_name, 
                         "train_res" : self.dat['train_dat'], "test_res" : self.dat['test_dat'], "scale_info" : self.scale_info, 
                         "nTimeSteps" : self.nTimeSteps, "nInterval" : self.nInterval})
        
        
    def _preprocess(self):
        # 전체 데이터 생성(시계열 설명변수 생성을 위해)
        self.full_dat = pd.concat([self.dat['train_dat'],self.dat['test_dat']], ignore_index=True)

        # LSTM 전처리 데이터 생성
        self.nTimeSteps = 5
        self.nInterval  = 1

        # 디멘전 배치 사이즈
        self.dim_batch = self.full_dat.shape[0] - (self.nInterval * (self.nTimeSteps - 1))

        # 데이터 인덱스 생성
        idx_list = []
        for i in range(self.dim_batch):
            idx_list.append(np.arange(start = i, stop = self.nInterval * (self.nTimeSteps - 1) + i + 1, step = self.nInterval))

        # LSTM 시계열 데이터 생성    
        x_array = []
        y_array = []
        date_df = []
        for idx in range(len(idx_list)):
            x_array.append(np.array(self.full_dat[self.xvar_name].iloc[idx_list[idx]]))
            y_array.append(np.array(self.full_dat[self.yvar_name].iloc[idx_list[idx]]))
            date_df.append(self.full_dat['predict_time'].iloc[idx_list[idx]].max())

        self.x_array = np.array(x_array)
        self.y_array = np.array(y_array)
        self.date_df = np.array(date_df)    

        # 학습, 검증 데이터로 다시 나누기
        
        self.test_start_date = self.dat['test_dat']['predict_time'].iloc[0]
        self.train_valid_x_array = self.x_array[self.date_df<self.test_start_date,:,:]
        self.train_valid_y_array = self.y_array[self.date_df<self.test_start_date,:]

        # Train Valid Split Point
        self.split_point = math.ceil(self.train_valid_x_array.shape[0] * 0.7)
        
        # Train
        self.train_x_array = self.train_valid_x_array[0:self.split_point,:,:]
        self.train_y_array = self.train_valid_y_array[0:self.split_point,:]

        # Valid
        self.valid_x_array = self.train_valid_x_array[self.split_point:,:,:]
        self.valid_y_array = self.train_valid_y_array[self.split_point:,:]
        
        # Test
        self.test_x_array = self.x_array[self.date_df>=self.test_start_date,:,:]
        self.test_y_array = self.y_array[self.date_df>=self.test_start_date,:]

        # Min, Max Scaling
        self.scale_info = pd.DataFrame({'xvar_name' : self.xvar_name, 
                                        'min' : np.apply_along_axis(min, 0, np.vstack(self.train_x_array)), 
                                        'max' : np.apply_along_axis(max, 0, np.vstack(self.train_x_array))})

        def _scaliling(xx):
            return (xx - self.scale_info['min'])/(self.scale_info['max'] - self.scale_info['min'] + 1e-10)
        
        # Min, Max Scaling
        self.x_tr_scale = np.apply_along_axis(_scaliling, 2, self.train_x_array)
        self.x_vd_scale = np.apply_along_axis(_scaliling, 2, self.valid_x_array)
        self.x_vd_scale[self.x_vd_scale<0] = 0
        self.x_vd_scale[self.x_vd_scale>1] = 1
        self.x_te_scale = np.apply_along_axis(_scaliling, 2, self.test_x_array)
        self.x_te_scale[self.x_te_scale<0] = 0
        self.x_te_scale[self.x_te_scale>1] = 1

            
    def _fit(self):
         
        # LSTM Build
        input_shape = (self.nTimeSteps, self.train_x_array.shape[2]) # input shape 설정
        
        if self.TuneMethod == 'BO' :
            print('='*40)
            print('Tuning Method : Bayesian Optimization'.center(40))    
            print('='*40)
            # BasianOptimization Tuning 
            tuner = keras_tuner.BayesianOptimization(lstm_model(input_shape), 
                                                 objective = 'val_loss', 
                                                 max_trials=10, # 튜닝 파라미터 시도 회수
                                                 num_initial_points=2,
                                                 executions_per_trial=2, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        if self.TuneMethod == 'HB' :
            print('='*40)
            print('Tuning Method : Hyper Band'.center(40))
            print('='*40)
            # HyperBand Tuning    
            tuner = keras_tuner.Hyperband(lstm_model(input_shape), 
                                                 objective = 'val_loss', 
                                                 max_epochs=5, 
                                                 factor = 3,
                                                 executions_per_trial=2, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        
        TryNum = 10    
        for i in range(TryNum):
            try :
                print(f'Tuner Search Try - {str(i).zfill(2)}')
                tuner.search(self.x_tr_scale, 
                             self.train_y_array, 
                             epochs = 10,
                             validation_data = (self.x_vd_scale, self.valid_y_array), # validation data를 입력해 줄 수 있음 (fit 함수와 동일)
                             verbose = 0,
                             batch_size = 16)
            except :
                # 에러 발생 시 다음 Try 시도
                continue
           
            # 에러 발생 하지 않을 시 루프 종료
            break


        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print(best_hps.values)
        self.model = tuner.hypermodel.build(best_hps)
        self.model.summary()

        # Early Stopping 
        """
        - restore_best_weights -
        Whether to restore model weights from the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of training are used
        """
        self.EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10,
                                                         restore_best_weights=True, verbose = 1) 

        # reduce_lr
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='min', min_lr=1e-5)


        # # 모델 Check Point 저장
        # ModelCheck = tf.keras.callbacks.ModelCheckpoint(filepath = 'D:/workspace/OUT/best_model_gru/best_model_gru_{}.h5'.format(self.model_num)
        #                                                 , monitor = 'val_loss'
        #                                                 , mode = 'min'
        #                                                 , save_best_only = True)

        # History
        self.history = tf.keras.callbacks.History()

        # 에폭 10까지 학습
        self.model.fit(self.x_tr_scale, self.train_y_array, epochs = 10, validation_data = (self.x_vd_scale, self.valid_y_array)
                       , batch_size = 16
                       , callbacks = [self.history]
                       , verbose=0)


        # 에폭 10부터 학습
        self.model.fit(self.x_tr_scale, self.train_y_array
                      , validation_data = (self.x_vd_scale, self.valid_y_array)
                      , initial_epoch = 10
                      , epochs = 100
                      , batch_size = 16
                      , callbacks = [self.EarlyStopping, self.reduce_lr, self.history]
                      , verbose=0)
        
        self.train_loss = self.model.history.history['loss']
        self.valid_loss = self.model.history.history['val_loss']
        self.epochs = self.model.history.epoch

    def _pred(self):
        self.tr_pred = self.model.predict(np.vstack([self.x_tr_scale,self.x_vd_scale]), verbose = 0)
        self.te_pred = self.model.predict(self.x_te_scale, verbose = 0)

        self.tr_pred = np.array([self.tr_pred[s][-1] for s in range(len(self.tr_pred))])
        self.tr_pred = np.vstack([np.zeros(self.nInterval * (self.nTimeSteps - 1)).reshape(-1,1),self.tr_pred])
        self.te_pred = np.array([self.te_pred[s][-1] for s in range(len(self.te_pred))])
        self.dat['train_dat']['pred'] = self.tr_pred
        self.dat['test_dat']['pred']  = self.te_pred 
        
        
        
## LSTM Model Class
class lstm_model(HyperModel):
    def __init__(self, input_shape : tuple):
        self.input_shape = input_shape

    def build(self, hp):
        input_layer = tf.keras.Input(shape = self.input_shape, name = 'input_layer')
        for x in range(hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)):
            if x == 0:
                lstm_layer = tf.keras.layers.LSTM(hp.Int('units_{}'.format(str(x+1)), min_value = 4, max_value = 32, step = 4)
                                                   , return_sequences=True, name = f'lstm_layer_{str(x+1)}')(input_layer)
            else :
                lstm_layer = tf.keras.layers.LSTM(hp.Int('units_{}'.format(str(x+1)), min_value = 4, max_value = 32, step = 4)
                                                   , return_sequences=True, name = f'lstm_layer_{str(x+1)}')(lstm_layer)
        output_layer = tf.keras.layers.Dense(1, activation = 'linear', kernel_initializer = tf.keras.initializers.orthogonal(seed = 1234), name = 'output_layer')(lstm_layer)  
        model = tf.keras.Model(inputs = input_layer, outputs = output_layer)
        
        # compile 
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'mse')  
        return model
    
    
'''
CNN Model Class
'''
class MODELING_CNN():
    def __init__(self, train_x : np.array , train_y : np.array , valid_x : np.array , valid_y : np.array , TuneMethod : str) :
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.TuneMethod   = TuneMethod
        self.input_shape  = train_x.shape[1:]
        self.output_shape = train_y.shape[-1]
        self._fit()
        
    def _fit(self):
        if self.TuneMethod == 'BO' :
            print('='*40)
            print('Tuning Method : Bayesian Optimization'.center(40))    
            print('='*40)
            # BasianOptimization Tuning 
            tuner = keras_tuner.BayesianOptimization(cnn_model(self.input_shape,self.output_shape), 
                                                 objective = 'val_loss', 
                                                 max_trials=5, # 튜닝 파라미터 시도 회수
                                                 num_initial_points=2,
                                                 executions_per_trial=1, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        if self.TuneMethod == 'HB' :
            print('='*40)
            print('Tuning Method : Hyper Band'.center(40))
            print('='*40)
            # HyperBand Tuning    
            tuner = keras_tuner.Hyperband(cnn_model(self.input_shape,self.output_shape), 
                                                 objective = 'val_loss', 
                                                 max_epochs=5, 
                                                 factor = 3,
                                                 executions_per_trial=1, # The number of models that should be built and fit for each trial.
                                                 directory='.',
                                                 seed = 1234,
                                                 project_name="hp_temp_folder",
                                                 overwrite = True)
        
        TryNum = 10    
        for i in range(TryNum):
            try :
                print(f'Tuner Search Try - {str(i).zfill(2)}')
                tuner.search(self.train_x, 
                             self.train_y, 
                             epochs = 10,
                             validation_data = (self.valid_x, self.valid_y), # validation data를 입력해 줄 수 있음 (fit 함수와 동일)
                             verbose = 0,
                             batch_size = 32)
            except :
                # 에러 발생 시 다음 Try 시도
                continue
           
            # 에러 발생 하지 않을 시 루프 종료
            break
            
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print(best_hps.values)
        self.model = tuner.hypermodel.build(best_hps)
        self.model.summary()
        
        # Early Stopping 
        """
        - restore_best_weights -
        Whether to restore model weights from the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of training are used
        """
        self.EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10,
                                                         restore_best_weights=True, verbose = 1) 

        # reduce_lr
        self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=0, mode='min', min_lr=1e-5)


        # # 모델 Check Point 저장
        # ModelCheck = tf.keras.callbacks.ModelCheckpoint(filepath = 'D:/workspace/OUT/best_model_gru/best_model_gru_{}.h5'.format(self.model_num)
        #                                                 , monitor = 'val_loss'
        #                                                 , mode = 'min'
        #                                                 , save_best_only = True)

        # History
        self.history = tf.keras.callbacks.History()
        
        
        # 에폭 10까지 학습
        self.model.fit(self.train_x, self.train_y, epochs = 10, validation_data = (self.train_x, self.train_y)
                       , batch_size = 32
                       , callbacks = [self.history]
                       , verbose=1)


        # 에폭 10부터 학습
        self.model.fit(self.train_x, self.train_y
                      , validation_data = (self.train_x, self.train_y)
                      , initial_epoch = 10
                      , epochs = 100
                      , batch_size = 32
                      , callbacks = [self.EarlyStopping, self.reduce_lr, self.history]
                      , verbose=1)
        
        self.train_loss = self.model.history.history['loss']
        self.valid_loss = self.model.history.history['val_loss']
        self.epochs = self.model.history.epoch
        
## Cnn Model Class
class cnn_model(HyperModel):
    def __init__(self, input_shape : tuple, output_shape : tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def build(self, hp):
        sInit = hp.Choice("initializer", ["random_normal","glorot_normal","he_normal","orthogonal"])
        
        # Initializer
        if sInit == 'random_normal' :
            initializer = tf.keras.initializers.random_normal(seed=1234)
        if sInit == 'glorot_normal' :
            initializer = tf.keras.initializers.glorot_normal(seed=1234)
        if sInit == 'he_normal' :
            initializer = tf.keras.initializers.he_normal(seed=1234)
        if sInit == 'orthogonal' :
            initializer = tf.keras.initializers.orthogonal(seed=1234)

        # Cnn layer    
        input_layer  = tf.keras.Input(shape = self.input_shape, name = 'input_layer')
        for x in range(hp.Int('num_layers', min_value = 1, max_value = 3, step = 1)):
            if x == 0:
                cnn_layer = tf.keras.layers.Conv2D(hp.Choice('units_{}'.format(str(x+1)), [16,32,128,256])
                                                   , kernel_size = 3, padding = 'same', strides = 1, name = f'cnn_layer_{str(x+1)}', kernel_initializer = initializer)(input_layer)
                cnn_layer = tf.keras.layers.BatchNormalization( name = f'conv{str(x+1)}_nor')(cnn_layer)
                cnn_layer = tf.keras.layers.Activation('relu', name = f'conv{str(x+1)}_act')(cnn_layer)
                if hp.Boolean('dropout_{}'.format(str(x+1))):
                    cnn_layer = tf.keras.layers.Dropout(0.25, name = f'conv{str(x+1)}_drop')(cnn_layer)
                cnn_layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), name = f'conv{str(x+1)}_pool')(cnn_layer)
            else :
                cnn_layer = tf.keras.layers.Conv2D(hp.Choice('units_{}'.format(str(x+1)), [16,32,128,256])
                                                   , kernel_size = 3, padding = 'same', strides = 1, name = f'cnn_layer_{str(x+1)}', kernel_initializer = initializer)(cnn_layer)
                cnn_layer = tf.keras.layers.BatchNormalization( name = f'conv{str(x+1)}_nor')(cnn_layer)
                cnn_layer = tf.keras.layers.Activation('relu', name = f'conv{str(x+1)}_act')(cnn_layer)
                if hp.Boolean('dropout_{}'.format(str(x+1))):
                    cnn_layer = tf.keras.layers.Dropout(0.25, name = f'conv{str(x+1)}_drop')(cnn_layer)
                cnn_layer = tf.keras.layers.MaxPool2D(pool_size=(3, 3), name = f'conv{str(x+1)}_pool')(cnn_layer)       

        faltten = tf.keras.layers.Flatten()(cnn_layer)
        dense = tf.keras.layers.Dropout(hp.Float("dense_drop1", min_value=0, max_value=0.5, step=0.1, default=0.5), name = "dense_drop1")(faltten)
        dense = tf.keras.layers.Dense(hp.Choice('dense1_units', [16,32,128,256]), activation = 'relu', kernel_initializer = initializer, name = 'dense1')(dense)
        dense = tf.keras.layers.Dropout(hp.Float("dense_drop2", min_value=0, max_value=0.5, step=0.1, default=0.5), name = "dense_drop2")(dense)
        output_layer = tf.keras.layers.Dense(self.output_shape, activation = 'softmax', kernel_initializer = initializer
                                             , kernel_regularizer=tf.keras.regularizers.l2(l2 = hp.Float("l2", min_value=1e-2, max_value=1e-1, sampling="log"))
                                             , name = 'dense2')(dense)                                     
        # Cnn Model
        cnn_model = tf.keras.Model(inputs = input_layer, outputs = output_layer)                      

        # Model Compile 
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss = 'categorical_crossentropy', metrics = 'acc')
        return cnn_model
