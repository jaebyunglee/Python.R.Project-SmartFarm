#==============================================================================|
# Title     : (스마트팜) 악취 예측모델 예측 함수 모음
# Sub-title : pred_prediction_tools
# Author    : Begas - 
# Updates   : 
#==============================================================================|

#------------------------------------------------------------------------------|
#------------ Step 0. Functions -----------------------------------------------
#------------------------------------------------------------------------------|

#-- Model Load Function
# working_date : 모델 학습 일자
# model_path   : 모델 위치
# yvar_name    : y 변수 명
# all.x.na     : x변수 전부 NA 여부
MODEL_LOAD_FN <- function(working_date, model_path, yvar_name, all.x.na = F){
  
  # 모든 x가 NA는 아닐 시
  if(all.x.na == F){
    
    best_model_path <- paste(model_path,working_date,yvar_name,'BEST_MODEL/',sep = '/')
    best_model_name <- strsplit(list.files(best_model_path)[1], ".", fixed = T)[[1]][1]
    
    if(best_model_name %in% c('LSTM','LSTM_Y','ANN')){
      load(paste0(best_model_path,best_model_name,'.RData'))
      model <- keras::load_model_hdf5(paste0(best_model_path,best_model_name,'.h5'))
      cat('Model Dir :', paste0(best_model_path,best_model_name,'.h5'),'\n')
    } else if (best_model_name == 'LIGHTGBM'){
      load(paste0(best_model_path,best_model_name,'.RData'))
      model <- lightgbm::lgb.load(paste0(best_model_path,best_model_name,'.LGB'))
      cat('Model Dir : ', paste0(best_model_path,best_model_name,'.LGB'),'\n')
    } else {
      load(paste0(best_model_path,best_model_name,'.RData'))
      model <- RESULT$model
      cat('Model Dir :', paste0(best_model_path,best_model_name,'.RData'),'\n')
    }
    
  # 모든 x가 NA일 시(Y만 사용한 모델 불러오기)  
  } else {
    
    onlyy_model_path <- paste(model_path,working_date,yvar_name,'MODEL/',sep = '/')
    load(paste(onlyy_model_path,'EVAL_RESULT.RData', sep = '/'))
    model_name <- EVAL_DF[EVAL_DF$model %in% c('LSTM_Y','ARIMAX_Y'),"model"][1]
    
    if(model_name %in% c('LSTM_Y')){
      load(paste0(onlyy_model_path,model_name,'.RData'))
      model <- keras::load_model_hdf5(paste0(onlyy_model_path,model_name,'.h5'))
      cat('Model Dir :', paste0(onlyy_model_path,model_name,'.h5'),'\n')
    } else {
      load(paste0(onlyy_model_path,model_name,'.RData'))
      model <- RESULT$model
      cat('Model Dir :',  paste0(onlyy_model_path,model_name,'.RData'),'\n')
    }
  }

  return(list(model = model, MODEL_RESULT = RESULT))
}

#-- Predict Function
# PREDICT_DATA : PRED_ANAL_FN의 출력
# MODEL_LOAD   : MODEL_LOAD_FN 출력
PRED_FN <- function(PREDICT_DATA, MODEL_LOAD) {

  model      <- MODEL_LOAD$model
  model_xvar <- MODEL_LOAD$MODEL_RESULT$xvar
  model_type <- MODEL_LOAD$MODEL_RESULT$model_type
  scale_info <- MODEL_LOAD$MODEL_RESULT$scale_info
  
  # 모델 예측
  predict_data <- PREDICT_DATA[, c('base_time', 'predict_time')]
  if (model_type %in% c('MLR', 'RF')) {
    pred <- predict(model, newdata = PREDICT_DATA)
    predict_data$pred_val <- pred
    
  } else if (model_type == 'XGBOOST') {
    # 전처리
    tmp_df <- PREDICT_DATA[, model_xvar]
    tmp_df <- xgb.DMatrix(data = as.matrix(tmp_df))
    
    pred <- predict(model, tmp_df)
    predict_data$pred_val <- pred
    
  } else if (model_type == 'LIGHTGBM') {
    # 전처리
    tmp_df <- PREDICT_DATA[, model_xvar]
    
    pred <- predict(model, as.matrix(tmp_df))
    predict_data$pred_val <- pred
    
  } else if (model_type == 'ANN') {
    tmp_df <- PREDICT_DATA[, rownames(scale_info)]
    
    # min max scale
    ann_minmax_scale <- function(data, scale_info){
      max <- scale_info['max'][,1]
      min <- scale_info['min'][,1]
      eps <- 1e-7
      ret <- apply(data,1, function(x){(x-min)/(max - min + eps)})
      return(t(ret))
    }
    
    tmp_df <- ann_minmax_scale(tmp_df, scale_info)
    
    pred <- model %>% predict(tmp_df)
    predict_data$pred_val <- pred
    
  } else if (model_type %in% c('LSTM','LSTM_Y')) {
    tmp_df <- PREDICT_DATA[, rownames(scale_info)]
    
    nTimeStep <- 5
    dim_batch <- dim(tmp_df)[1] - 1 * (nTimeStep - 1)
    idx_array <- sapply(1:dim_batch, function(s) {
      seq(from = s, to = 1 * (nTimeStep - 1) + s, by = 1)
    })
    
    # Lstm Data 생성
    lstm_array_x <- array(NA, dim = c(dim_batch, nTimeStep, dim(tmp_df)[2]))
    
    for(idx in 1:dim_batch) {
      lstm_array_x[idx,,] <- as.matrix(tmp_df[idx_array[, idx],])
    }
    
    # t시점의 prediction time 가져오기
    predict_data <- predict_data[idx_array[nTimeStep, ], ]
    
    # Scaling
    lstm_minmax_scale <- function(data, scale_info){
      max <- scale_info['max'][,1]
      min <- scale_info['min'][,1]
      eps <- 1e-7
      for(idx in 1:dim(data)[1]){
        data[idx,,] <- t(apply(data[idx,,], 1 ,function(x){(x - min)/(max - min + eps)}))
      }
      return(data)
    }
    
    lstm_array_x <- lstm_minmax_scale(lstm_array_x, scale_info)
    
    # Predict
    pred <- model %>% predict(lstm_array_x)
    pred <- pred[, dim(pred)[2], ]
    predict_data$pred_val <- pred
    
  } else if (model_type %in% c('ARIMAX','ARIMAX_Y')) {
    if(is.null(model_xvar)){
      
      pred <- forecast::forecast(model, h = dim(PREDICT_DATA)[1])$mean
      predict_data$pred_val <- pred
      
    } else {
      
      tmp_df <- PREDICT_DATA[, model_xvar]
      pred <- forecast::forecast(model, xreg = as.matrix(tmp_df))$mean
      predict_data$pred_val <- pred
    }
    
  }
  
  return(predict_data)
}