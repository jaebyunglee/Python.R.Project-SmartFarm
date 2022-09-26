#==============================================================================|
# Title     : (스마트팜) 악취 예측모델 예측 코어
# Sub-title : Prediction Data Preprocess & Prediction
# Author    : Begas - 
# Updates   : 
#==============================================================================|
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
source(paste(home_path,"5. SRC/pred_prediction_tools.R",sep = "/"))

# 4. Model training date 설정 (train core의 working_date)
training_date <- '20220914'

# 5. 예측할 데이터의 base_date 설정
cutoff_date <- "2021-08-10 13:00:00"
cutoff_date <- format(as.POSIXct(cutoff_date), "%Y-%m-%d %H:%M:%S")

#------------------------------------------------------------------------------|
#------------ Step 2. Read Dataset --------------------------------------------
#------------------------------------------------------------------------------|

# 1. 데이터 불러오기
org_dat <- LOAD_FN(data_files)
str(org_dat)

# 2. 모든 X 변수 NA 여부 체크
variables <- c('온도','습도','환기팬','거품도포시간','거품도포량')
variables <- colnames(org_dat)[colnames(org_dat) %in% variables]
all.x.na  <- length(variables) == 0

# 모든 X 변수가 NA일 시 황화수소와 암모니아의 과거값으로 예측
if(all.x.na){
  if(mean(c("황화수소","암모니아") %in% colnames(org_dat)) != 1){
    stop("모든 설명변수가 NA일 시 황화수소와 암모니아의 데이터가 있어야만 합니다. 모델 예측을 중단합니다.")
  }
}
#------------------------------------------------------------------------------|
#------------ Step 3. Data Preprocess -----------------------------------------
#------------------------------------------------------------------------------|

# 1. 데이터 전처리1 (이상치/결측치 처리/시간대별 요약테이블 생성)
DAT <- PREPROCESS_FN(org_dat, time_grp = 60)

# 2. 분석 데이터 생성 (설명변수 시점이전 및 검증데이터 생성)
ANAL_DAT <- PRED_ANAL_FN(DAT, window_v = c(12, 24))

#------------------------------------------------------------------------------|
#------------ Step 4. Prediction -----------------------------------------------
#------------------------------------------------------------------------------|
model_y_type <- list.files(paste(model_path,training_date,sep = '/'))

RESULT <- list()
for(yvar_name in model_y_type){
  # Best Model Load
  MODEL_LOAD <- MODEL_LOAD_FN(training_date, model_path, yvar_name = yvar_name, all.x.na = all.x.na)
  
  PRED_RESULT <- PRED_FN(ANAL_DAT$pred, MODEL_LOAD)
  PRED_RESULT$yvar_name <- yvar_name
  
  # base_date >= cutoff_date
  RESULT[[yvar_name]] <- PRED_RESULT[PRED_RESULT$base_time >= cutoff_date,]
}

FINAL_RESULT <- rbindlist(RESULT)
FINAL_RESULT

