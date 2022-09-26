#==============================================================================|
# Title     : (스마트팜) 악취 예측모델 학습 전처리 함수 모음
# Sub-title : common_preprocess_tools
# Author    : 이재병
# Updates   : 2022.09.26
#==============================================================================|

#------------------------------------------------------------------------------|
#------------ Step 0. Functions -----------------------------------------------
#------------------------------------------------------------------------------|


# 함수 1. 데이터 호출
# data_path : raw 데이터 위치
LOAD_FN <- function(data_path){
  cat("Data loading:", data_path, "\n")
  # Data load
  dat <- fread(data_path, encoding = 'UTF-8')
  
  # 데이터 변수명 변경 
  if(!is.null(dat$sensingDt)){colnames(dat)[colnames(dat) == 'sensingDt'] <- '시간'}
  if(!is.null(dat$nh3))      {colnames(dat)[colnames(dat) == 'nh3']       <- '암모니아'}
  if(!is.null(dat$h2s))      {colnames(dat)[colnames(dat) == 'h2s']       <- '황화수소'}
  if(!is.null(dat$tmp))      {colnames(dat)[colnames(dat) == 'tmp']       <- '온도'}
  if(!is.null(dat$hum))      {colnames(dat)[colnames(dat) == 'hum']       <- '습도'}
  if(!is.null(dat$voc))      {colnames(dat)[colnames(dat) == 'voc']       <- '환기팬'}
  
  # Time : Text to POSIXct
  dat$시간 <- as.POSIXct(dat$시간)
  
  # 제거1. 암모니아와 황화수소가 두개의 변수일 경우 ppm으로 선택
  if(sum(grepl("암모니아", names(dat))) > 1){
    candi.cols <- names(dat)[grepl("암모니아", names(dat))]
    del.col <- candi.cols[grepl("(㎷)|mV", candi.cols)] 
    dat[[del.col]] <- NULL
  }
  if(sum(grepl("황화수소", names(dat))) > 1){
    candi.cols <- names(dat)[grepl("황화수소", names(dat))]
    del.col <- candi.cols[grepl("(㎷)|mV", candi.cols)] 
    dat[[del.col]] <- NULL
  }
  
  # 제거2. 변수명에 영어 포함시 제거
  names(dat) <- gsub("[a-z]|[[:punct:]]|℃|㎷", "", names(dat))
  
  # 제거3. 모든 값이 NA인 변수 제거
  cat("Data preprocssing: 모든 값이 NA인 변수 제거\n")
  del.cols <- names(which(colMeans(is.na(dat)) == 1))
  if(length(del.cols)>0){cat(del.cols,'\n')}
  dat <- data.frame(dat)[, setdiff(names(dat), del.cols)]
  
  # 제거4. 단일값
  cat("Data preprocssing: 단일 값만 가지는 변수 제거\n")
  del.cols <- c()
  for(c in names(dat)){
    if(uniqueN(dat[[c]]) == 1){
      del.cols <- c(del.cols, c)
    }
  }
  if(length(del.cols)>0){cat(del.cols,'\n')}
  
  dat.new <- data.frame(dat)[, setdiff(names(dat), del.cols)]
  return(dat.new)
}

# 함수 2. 데이터 전처리 (이상치/결측치 처리 , 시간대별 요약테이블 생성)
# dat      : 데이터
# time_grp : 요약 시간(ex : if time_grp = 60, then 1시간 단위로 데이터 요약)  
PREPROCESS_FN <- function(dat, time_grp) {
  
  cat("Step1. 이상치를 허용범위 내로 보정\n")
  dat[(dat$암모니아     <   0) & (!is.na(dat$암모니아))    , '암모니아'    ] <- 0
  dat[(dat$황화수소     <   0) & (!is.na(dat$황화수소))    , '황화수소'    ] <- 0
  dat[(dat$환기팬       <   0) & (!is.na(dat$환기팬))      , '환기팬'      ] <- 0
  dat[(dat$거품도포량   <   0) & (!is.na(dat$거품도포량))  , '거품도포량'  ] <- 0
  dat[(dat$거품도포시간 <   0) & (!is.na(dat$거품도포시간)), '거품도포시간'] <- 0
  dat[(dat$온도         < -50) & (!is.na(dat$온도))        , '온도'        ] <- -50
  dat[(dat$온도         >  50) & (!is.na(dat$온도))        , '온도'        ] <-  50
  dat[(dat$습도         <   0) & (!is.na(dat$습도))        , '습도'        ] <-   0
  dat[(dat$습도         > 100) & (!is.na(dat$습도))        , '습도'        ] <- 100
  
  cat("Step2. 시간 변수를", time_grp, "분 단위로 변경\n")
  dat$newtime  <- lubridate::floor_date(dat$시간,unit = paste(as.character(time_grp), "minutes"))
  dat <- dat %>% select(-시간)
  
  cat("Step3. 시간(newtime)별 변수들(variable)의 값(value)을 갖는 형태로 변환\n")
  dat2 <- reshape2::melt(dat, id.vars = c('newtime'), na.rm = T) %>% rename(var1 = variable)
  
  cat("Step4. 각 variable별 요약통계량 변수 생성\n")
  dat3 <- dat2 %>% group_by(newtime, var1) %>%
    summarise(v_mean   = mean(value),v_median = median(value),
              v_min    = min(value) ,v_max    = max(value)   , v_std    = sd(value)) %>% data.frame()
  
  cat("Step5. 시간(newtime), 변수들(var1)별 요약통계량(var2)의 값(value)을 갖는 형태로 변환\n")
  dat4 <- reshape2::melt(dat3,id.vars = c('newtime', 'var1'),measure.vars = names(dat3)[grepl("v_", names(dat3))],
                         na.rm = T) %>% rename(var2 = variable)
  
  cat("Step6. 시간별 변수들의 요약통계량 데이터프레임으로 재구조화\n")
  dat5 <- dat4 %>% mutate(variable = paste(var1, var2, sep = '_')) %>%select(-var1, -var2)
  dat5 <- reshape2::acast(dat5, newtime ~ variable, value.var = 'value') %>% data.frame
  dat5$time <- as.POSIXct(rownames(dat5))
  rownames(dat5) <- NULL
  
  cat("Step7. 특정 시간대의 데이터가 비어있을 시 해당 시간대 생성\n")
  base_time_df <- data.frame(time = seq(dat5$time[1], dat5$time[length(dat5$time)],60*60))
  if(dim(base_time_df)[1] > dim(dat5)[1]){
    cat(dim(base_time_df)[1] - dim(dat5)[1],'개의 시간대가 누락되어 있습니다.\n')
    cat('==== 누락된 시간대 ====\n') ; print(head(as.POSIXct(setdiff(base_time_df$time,dat5$time), origin = '1970-01-01')))
    dat5 <- merge(base_time_df, dat5, all.x = T)
  }
  
  cat("Step8. 선형보간법 적용\n")
  linear_interpolation <- function(x) {
    which_not_na <- which(!is.na(x))
    if (length(which_not_na)) { #--->>> 전부 NA는 아닌 경우
      start_idx <- min(which_not_na)
      end_idx <- max(which_not_na)
    
      tmp_x <- x[start_idx:end_idx]
      tmp_x <- na.approx(tmp_x)
      
      x[start_idx:end_idx] <- tmp_x
    }
    return(x)
  }
  for (cn in colnames(dat5)) {
    if (cn == 'time') next
    dat5[[cn]] <- linear_interpolation(dat5[[cn]])
  }
  return(dat5)
}

# 함수 3. 학습 데이터 전처리 (종속변수, 설명변수 지정 / 시차 데이터 생성)
# dat      : PREPROCESS_FN의 아웃풋 데이터
# window_v : 시차 (예측시간으로부터 window_v시간 전의 데이터를 설명변수로 사용)
TRAIN_ANAL_FN <- function(dat, window_v = c(12, 24)) {
  cat("Step1. 목표변수: 암모니아, 황화수소 단위별 max 값\n")
  origin_dat <- dat
  
  dat1 <- dat %>% 
    mutate(predict_time = time, y_황화수소 = 황화수소_v_max,y_암모니아 = 암모니아_v_max) %>%
    select(predict_time, y_황화수소, y_암모니아) %>% arrange(predict_time)
  
  i = 1
  for (w in window_v) {
    cat("Step2-", i, ". 시차변수 생성 (window = ", w, ")\n", sep = "")
    dat1 <- dat1 %>% mutate(time = predict_time - hours(w))
    # 여기서 time은 window 시간 이전을 의미함.
    dat <- origin_dat
    colnames(dat) <- paste(colnames(dat), paste0(w, 'h'), sep = '_')
    colnames(dat)[grep('time_', colnames(dat))] <- 'time'
    dat1 <- merge(dat1, dat, by = 'time', all.x = T) %>% select(-time)
    i = i + 1
  }
  
  cat("Step3. 시차 데이터 생성으로 인한 결측치 제거\n")
  dat2 <- dat1[(max(window_v) + 1):nrow(dat1), ]
  
  
  cat("Step4. 데이터 분할 (Train : Test =  7 : 3)\n")
  train_df <- dat2[1:floor(nrow(dat2) * 0.7), ]
  test_df  <- dat2[(floor(nrow(dat2) * 0.7) + 1):nrow(dat2), ]
  
  cat("Step5. var_dict 생성\n")
  var_dict = list("time" = c("predict_time"),"target" = c("y_암모니아", "y_황화수소"))
  
  cat("Step6. 변수 제거(단일값)\n")
  model_var <- setdiff(colnames(train_df), c(var_dict[['time']], var_dict[['target']]))
  
  del_cols <- c()
  for(c in names(train_df)){
    if(uniqueN(train_df[[c]]) == 1){
      del_cols <- c(del_cols, c)
    }
  }
  if(length(del_cols) > 0){cat(del_cols, '\n')}
  
  model_var <- setdiff(model_var, del_cols)
  var_dict[['x']] <- model_var
  
  RESULT <- list("train" = train_df,"test" = test_df,"var_dict" = var_dict)
  
  return(RESULT)
}

# 함수 4. 예측 데이터 전처리 (종속변수, 설명변수 지정 / 시차 데이터 생성)
# dat      : PREPROCESS_FN의 아웃풋 데이터
# window_v : 시차 (예측시간으로부터 window_v시간 전의 데이터를 설명변수로 사용)
PRED_ANAL_FN <- function(dat, window_v = c(12, 24)) {
  cat("Step1. 예측시간 설정(현재시간 + window_v min 값)\n")
  origin_dat <- dat
  
  dat1 <- dat %>%
    mutate(predict_time = time + hours(min(window_v))) %>%
    select(predict_time) %>% arrange(predict_time)
  
  i = 1
  for (w in window_v) {
    cat("Step2-", i, ". 시차변수 생성 (window = ", w, ")\n", sep = "")
    dat1 <- dat1 %>% mutate(time = predict_time - hours(w))
    # 여기서 time은 window 시간 이전을 의미함.
    
    dat <- origin_dat
    colnames(dat) <- paste(colnames(dat), paste0(w, 'h'), sep = '_')
    colnames(dat)[grep('time_', colnames(dat))] <- 'time'
    
    dat1 <- merge(dat1, dat, by = 'time', all.x = T) %>% select(-time)
    
    i = i + 1
  }
  
  cat("Step3. 시차 데이터 생성으로 인한 결측치 제거\n")
  dat2 <- dat1[(max(window_v) + 1):nrow(dat1), ]
  
  cat("Step4. base_time 컬럼 생성 \n")
  dat2 <- dat2 %>% mutate(base_time = predict_time - hours(min(window_v)))
  RESULT <- list("pred" = dat2)
  return(RESULT)
}



