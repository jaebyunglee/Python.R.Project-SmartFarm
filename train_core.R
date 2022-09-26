#==============================================================================|
# Title     : (스마트팜) 악취 예측모델 학습 코어
# Sub-title : Training Data Preprocess & Modeling
# Author    : Begas - 
# Updates   : 
#==============================================================================|ㅛ
rm(list=ls())
gc()

#------------------------------------------------------------------------------|
#------------ Step 0. Functions -----------------------------------------------
#------------------------------------------------------------------------------|

# 함수 1. 패키지 호출
#  - 함수 내 오브젝트 packages에 패키지명 추가
load_packages <- function() {
  packages <- c('data.table', 'dplyr', 'zoo', 'lubridate', 'corrplot', 'rpart', 'randomForest','reshape2',
                'ggplot2', 'gridExtra', 'ranger', 'gbm', 'keras', 'xgboost', 'lightgbm', 'forecast','xlsx')
  
  load_yn <- sapply(packages, function(pkg) {
    # 미설치된 패키지 설치
    if (!pkg %in% rownames(installed.packages())) {
      install.packages(pkg, dependencies = T)
    }
    # 패키지 불러오기
    require(pkg, character.only = T)
  })
  
  res <- data.frame(package_name = packages, load_yn)
  return(res)
}


#------------------------------------------------------------------------------|
#------------ Step 1. Settings ------------------------------------------------
#------------------------------------------------------------------------------|
# 0. 로드 패키지, 시드설정
load_packages()
set.seed(2022)
tensorflow::set_random_seed(2022)

# 1. 작업경로 설정
setwd('C:/Users/begas/Desktop/Project/SmartFarm')
home_path   <- getwd()
data_path   <- paste(home_path, "1. DAT", sep="/")
save_path   <- paste(home_path, "2. OUT", sep="/")
model_path  <- paste(home_path, "3. MODEL", sep="/")

# 2. 데이터 경로 설정
file_name   <- '샘플 데이터.csv'
# file_name   <- list.files(data_path)[1]
data_files  <- paste(data_path, file_name, sep="/")

# 3. 함수 불러오기
source(paste(home_path,"5. SRC/common_preprocess_tools.R",sep = "/"))
source(paste(home_path,"5. SRC/train_model_tools.R",sep = "/"))

# 4. Working date 설정
working_date <- '20220914'
#------------------------------------------------------------------------------|
#------------ Step 2. Read Dataset --------------------------------------------
#------------------------------------------------------------------------------|
# 1. 데이터 불러오기
org_dat <- LOAD_FN(data_files)
str(org_dat)

# 2. 종속변수 체크
if(mean(c("황화수소","암모니아") %in% colnames(org_dat)) != 1){
  stop("황화수소 또는 암모니아의 데이터가 전부 NA거나 하나의 값만 가집니다. 모델 학습을 중단합니다.")
  }
#------------------------------------------------------------------------------|
#------------ Step 3. Data Preprocess -----------------------------------------
#------------------------------------------------------------------------------|

# 1. 데이터 전처리1 (이상치/결측치 처리/시간대별 요약테이블 생성)
DAT <- PREPROCESS_FN(org_dat, time_grp = 60)

# 2. 분석 데이터 생성 (설명변수 시점이전 및 검증데이터 생성)
ANAL_DAT <- TRAIN_ANAL_FN(DAT, window_v = c(12, 24))
#------------------------------------------------------------------------------|
#------------ Step 4. Modeling ------------------------------------------------
#------------------------------------------------------------------------------|

train_data <- ANAL_DAT$train
test_data  <- ANAL_DAT$test
setDF(train_data)
setDF(test_data)
xvar_name <- ANAL_DAT$var_dict$x
for(yvar_name in ANAL_DAT$var_dict$target){
  cat('Modeling - Y var :', yvar_name,'\n')
  
  # 1. 모델 저장 경로 생성
  unlink(paste(model_path, working_date, yvar_name, 'MODEL', sep = '/'), recursive = T)
  unlink(paste(model_path, working_date, yvar_name, 'BEST_MODEL', sep = '/'), recursive = T)
  dir.create(paste(model_path, working_date, yvar_name, 'MODEL', sep = '/'), recursive = T)
  dir.create(paste(model_path, working_date, yvar_name, 'BEST_MODEL', sep = '/'), recursive = T)
  
  # 2. Model Fit
  RESULT_MLR    <- MODELING_MLR(working_date, model_path, train_data, test_data, yvar_name, xvar_name)
  RESULT_RF     <- MODELING_RF(working_date, model_path, train_data, test_data, yvar_name, xvar_name)
  RESULT_XGB    <- MODELING_XGBOOST(working_date, model_path, train_data, test_data, yvar_name, xvar_name)
  RESULT_LGB    <- MODELING_LIGHTGBM(working_date, model_path,train_data, test_data, yvar_name, xvar_name)
  RESULT_ANN    <- MODELING_ANN(working_date, model_path,train_data, test_data, yvar_name, xvar_name)
  RESULT_LSTM   <- MODELING_LSTM(working_date, model_path,train_data, test_data, yvar_name, xvar_name)
  RESULT_ARIMAX <- MODELING_ARIMAX(working_date, model_path,train_data, test_data, yvar_name, xvar_name)
  
  # Y 데이터만 활용해서 모델 생성
  RESULT_LSTM_Y   <- MODELING_LSTM(working_date, model_path,train_data, test_data, yvar_name, xvar_name, only.y = T)
  RESULT_ARIMAX_Y <- MODELING_ARIMAX(working_date, model_path,train_data, test_data, yvar_name, xvar_name, only.y = T)
  
  # 3. Fit Result
  RESULT_ALL      <- list(RESULT_MLR, RESULT_RF, RESULT_XGB, RESULT_LGB, RESULT_ANN,
                          RESULT_LSTM, RESULT_ARIMAX,RESULT_LSTM_Y,RESULT_ARIMAX_Y)

  # 4. Model Evaluation
  thrs    <- ifelse(grepl('암모니아',yvar_name),70,0.2) # threshold 암모니아 70, 황화수소 0.2
  EVAL_DF <- EVAL_FN(RESULT_ALL, thrs = thrs, type = 'mape') # type : mape, acc
  print(EVAL_DF)
  
  # 5. Save Evaluation Result
  save(EVAL_DF, file = paste(model_path, working_date, yvar_name, 'MODEL','EVAL_RESULT.RData', sep = '/'))
  
  # 6. Best Model Save
  SAVE_FN(working_date, model_path, yvar_name, EVAL_DF)
  
  # 7. 데이터 성능 저장
  eval_file_name = paste(save_path,stringr::str_replace(file_name,'.csv','_Eval.xlsx'), sep = '/')
  eval = tryCatch(write.xlsx(EVAL_DF, file = eval_file_name, row.names = F, sheetName = yvar_name, append = T)
                  ,error = function(e){"err"})
  if(is.character(eval)){
    unlink(eval_file_name)
    write.xlsx(EVAL_DF, file = eval_file_name, row.names = F, sheetName = yvar_name, append = T)
  }
 
}
