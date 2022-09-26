import os
import re
import math
import copy
import datetime
import numpy as np
import pandas as pd


'''
공통 전처리 클래스
'''

class common_preprocess():
    
    def __init__(self, data_files : os.path, time_grp : int, time_window : list):
        self.time_grp    = time_grp     # 시간대별 통계량 요약 기준 단위(ex : if time_grp = 60, then 60분 단위로 통계량 요약)
        self.data_files  = data_files   # Raw 데이터 위치
        self.time_window = time_window  # 시차 (예측시간으로부터 window_v시간 전의 데이터를 설명변수로 사용)

    '''
    - raw 데이터 호출 및 단일값 컬럼 제거
    '''
    def load_fn(self):
        '''
        * 입력
        data_files : raw 데이터 위치

        * 출력
        raw_dat    : 데이터 프레임
        '''
        print('>> Raw 데이터 호출 및 단일값 컬럼 제거')
        
        # 데이터 불러오기
        self.dat = pd.read_csv(self.data_files)

        # 데이터 변수명 변경
        if 'sensingDt' in self.dat.columns : self.dat.rename(columns = {'sensingDt' : '시간'    }, inplace = True)
        if 'nh3'       in self.dat.columns : self.dat.rename(columns = {'nh3'       : '암모니아'}, inplace = True)
        if 'h2s'       in self.dat.columns : self.dat.rename(columns = {'h2s'       : '황화수소'}, inplace = True)
        if 'tmp'       in self.dat.columns : self.dat.rename(columns = {'tmp'       : '온도'    }, inplace = True)
        if 'hum'       in self.dat.columns : self.dat.rename(columns = {'hum'       : '습도'    }, inplace = True)
        if 'voc'       in self.dat.columns : self.dat.rename(columns = {'voc'       : '환기팬'  }, inplace = True)

        # Time : date 변수로 변경
        self.dat['시간'] = pd.to_datetime(self.dat['시간'])

        # 제거1. 암모니아와 황화수소가 두개의 변수일 경우 ppm으로 선택
        if len([s for s in self.dat.columns if '암모니아' in s]) > 1 : 
            del_cols = [s for s in self.dat.columns if ('mV' in s) or ('(㎷)' in s)]
            self.dat = self.dat.drop(columns = del_cols)

        if len([s for s in self.dat.columns if '황화수소' in s]) > 1 : 
            del_cols = [s for s in self.dat.columns if ('mV' in s) or ('(㎷)' in s)]
            self.dat = self.dat.drop(columns = del_cols)

        # 제거2. 변수명에 영어 포함시 영어 제거
        cols_list = []
        for cols in self.dat.columns:
            result = re.sub("[a-zA-Z]|[^\w\s]", "", cols)
            cols_list.append(result)
        self.dat.columns = cols_list

        # 제거3. 모든 값이 NA인 변수 제거
        del_cols = self.dat.columns[self.dat.isna().mean() == 1]
        if len(del_cols) > 0 :
            print('모든 값이 NA인 변수 제거 :','/'.join(del_cols))
            self.dat = self.dat.drop(columns = del_cols)

        # 제거4. 단일값
        col_unq_val = self.dat.apply(lambda xx : len(xx.unique()), axis = 0)
        del_cols    = self.dat.columns[col_unq_val == 1]
        if len(del_cols) > 0 :
            print('모든 값이 단일값인 변수 제거 :','/'.join(del_cols))
            self.dat = self.dat.drop(columns = del_cols) 

        return self.dat
    
    def preprocess_fn(self) -> pd.DataFrame :
        '''
        * 입력
        dat              : LOAD_FN의 출력 데이터 프레임
        time_grp         : 시간대별 통계량 요약 기준 단위(ex : if time_grp = 60, then 60분 단위로 통계량 요약)

        * 출력
        final_summary_df : 데이터 프레임
        '''
        
        print('>> 이상치, 결측치 처리 및 시간대별 요약 통계량 생성')
        
        dat = self.dat.copy()
        
        print("Step1. 이상치를 허용범위 내로 보정")
        if '환기팬'       in dat.columns : dat.loc[( dat['환기팬']       <   0 ) & (~dat['환기팬'].isna())      ,'환기팬']       =   0
        if '암모니아'     in dat.columns : dat.loc[( dat['암모니아']     <   0 ) & (~dat['암모니아'].isna())    ,'암모니아']     =   0
        if '황화수소'     in dat.columns : dat.loc[( dat['황화수소']     <   0 ) & (~dat['황화수소'].isna())    ,'황화수소']     =   0
        if '거품도포량'   in dat.columns : dat.loc[( dat['거품도포량']   <   0 ) & (~dat['거품도포량'].isna())  ,'거품도포량']   =   0
        if '거품도포시간' in dat.columns : dat.loc[( dat['거품도포시간'] <   0 ) & (~dat['거품도포시간'].isna()),'거품도포시간'] =   0

        if '온도' in dat.columns : dat.loc[( dat['온도'] >  50 ) & (~dat['온도'].isna()),'온도'] =  50
        if '온도' in dat.columns : dat.loc[( dat['온도'] < -50 ) & (~dat['온도'].isna()),'온도'] = -50
        if '습도' in dat.columns : dat.loc[( dat['습도'] > 100 ) & (~dat['습도'].isna()),'습도'] = 100
        if '습도' in dat.columns : dat.loc[( dat['습도'] <   0 ) & (~dat['습도'].isna()),'습도'] =   0

        print("Step2. 시간 변수를", self.time_grp, "분 단위로 변경")

        def floor_dt(dt):
            # how many secs have passed this day
            nsecs = dt.hour*3600 + dt.minute*60 + dt.second + dt.microsecond*1e-6
            delta = nsecs % (self.time_grp * 60)
            return dt - datetime.timedelta(seconds=delta)

        dat['시간'] = dat['시간'].apply(floor_dt)  

        print("Step3. 시간별 요약통계량 데이터 생성")
        summary_cols = [s for s in dat.columns if '시간' not in s]
        mean_df = dat.groupby('시간').apply(lambda xx : xx[summary_cols].mean(skipna = True)).reset_index(drop = False).rename(columns = {s:s+"_mean" for s in summary_cols})
        min_df  = dat.groupby('시간').apply(lambda xx : xx[summary_cols].min(skipna = True)).reset_index(drop = True).rename(columns = {s:s+"_min" for s in summary_cols})
        max_df  = dat.groupby('시간').apply(lambda xx : xx[summary_cols].max(skipna = True)).reset_index(drop = True).rename(columns = {s:s+"_max" for s in summary_cols})
        std_df  = dat.groupby('시간').apply(lambda xx : xx[summary_cols].std(skipna = True)).reset_index(drop = True).rename(columns = {s:s+"_std" for s in summary_cols})

        # 요약 통계량 데이터 프레임 생성
        summary_df = pd.concat([mean_df,min_df,max_df,std_df], axis = 1)

        print("Step4. 특정 시간대의 데이터가 비어있을 시 해당 시간대 생성")
        st_time = summary_df['시간'].iloc[0]
        ed_time = summary_df['시간'].iloc[-1]
        base_date_df = pd.DataFrame({'시간' : pd.date_range(st_time,ed_time, freq = 'H')})
        n_missing_hour = len(set(set(base_date_df['시간'])) - set(set(summary_df['시간'])))
        print(f'- {n_missing_hour}개의 시간대가 비어있습니다. 해당 시간대를 생성합니다.')

        print("Step5. 선형 보간법 적용")
        self.final_summary_df = pd.merge(base_date_df,summary_df, how = 'left', on = '시간')
        self.final_summary_df.iloc[:,1:] = self.final_summary_df.iloc[:,1:].interpolate(method='linear')

        return self.final_summary_df
    
    def train_anal_dat_fn(self) -> dict :
        '''
        * 입력
        final_summary_df : PREPROCESS_FN의 출력 데이터 프레임

        * 출력
        final_summary_df : 데이터 프레임
        '''
        
        print('>> 학습 - 분석용 데이터 생성[종속, 설명(시차변수) 생성]')
        
        dat = self.final_summary_df
        
        print("Step1. 목표변수: 암모니아, 황화수소 단위별 max 값")
        dat.rename(columns = {'암모니아_max' : 'y_암모니아', '황화수소_max' : 'y_황화수소','시간' : 'base_time'}, inplace = True)
        dat.rename(columns = {s : 'x_' + s for s in dat.columns if ('time' not in s) and ('y_' not in s)}, inplace = True)

        print("Step2. 시차 변수 생성")
        old_dat = dat.copy()
        new_dat = dat.copy()

        new_dat = new_dat[['base_time'] + [s for s in new_dat.columns if 'y_' in s]].copy()
        new_dat.rename(columns = {'base_time' : 'predict_time'}, inplace = True)

        xvar_list = [s for s in old_dat.columns if 'x_' in s]
        for w in self.time_window :
            new_dat['base_time'] =\
            new_dat['predict_time'].apply(lambda xx : pd.date_range(end = xx, periods = w + 1, freq = 'H')[0])

            new_dat = \
            pd.merge(new_dat, old_dat[['base_time'] + xvar_list].rename(columns = {s : s +'_'+str(w) for s in xvar_list})
                    , how = 'left'
                    , on  = 'base_time')

        # 시차 데이터로 인한 결측 제거
        new_dat = new_dat.iloc[max(self.time_window):,:].reset_index(drop = True)

        print("Step3. 학습/검증 데이터 7:3으로 분할")
        train_dat = new_dat.iloc[:math.ceil(new_dat.shape[0] * 0.7),:].reset_index(drop = True)
        test_dat  = new_dat.iloc[math.ceil(new_dat.shape[0] * 0.7):,:].reset_index(drop = True)

        print("Step4. 단일 값만 가지는 설명변수 제거")
        # 학습 데이터에서 단일값만 가지는 변수 제거
        xvar_list = [s for s in train_dat.columns if 'x_' in s]
        single_value_var_index = np.where(train_dat[xvar_list].apply(lambda xx : xx.nunique()) == 1)[0]
        single_value_var_names = [xvar_list[s] for s in single_value_var_index]
        xvar_list = list(set(xvar_list) - set(single_value_var_names))
        xvar_list.sort()

        self.train_anal_dat = dict({'train_dat' : train_dat, 'test_dat' : test_dat, 'x_var' : xvar_list})
        return self.train_anal_dat