#==============================================================================|
# Title     : (스마트팜) 악취 예측모델 학습 모델링 함수 모음
# Sub-title : train_model_tools
# Author    : 이재병
# Updates   : 2022.09.26
#==============================================================================|

#------------------------------------------------------------------------------|
#------------ Step 0. Functions -----------------------------------------------
#------------------------------------------------------------------------------|

#-- 함수 1. Modeling Function
# working_date : 학습 일자
# model_path   : 모델 저장 위치  
# train_data   : ANAL_FN의 결과 - 학습 데이터
# test_data    : ANAL_FN의 결과 - 검증 데이터
# yvar_name    : y 변수 명
# xvar_name    : x 변수 명
# only.y       : y 데이터만으로 모델링

MODELING_MLR      <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name){
  cat("Model Type is MLR \n")
  
  # Fit
  formula <- paste0(yvar_name,' ~ ', paste(xvar_name, collapse = "+"))
  mlr_model <- lm(formula, data = train_data)
  summary(mlr_model)
  
  # Pred
  train_data$pred  <- predict(mlr_model, newdata = train_data)
  test_data$pred  <- predict(mlr_model, newdata = test_data)
  
  # Result
  RESULT <- list('working_date' = working_date,
                 'model_type' = 'MLR',
                 'yvar'       = yvar_name,
                 'xvar'       = xvar_name,
                 'model'      = mlr_model,
                 'train_res'  = train_data,
                 'test_res'   = test_data)
  
  # Save
  save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'MLR.RData', sep = '/'))
  
  return(RESULT)
}


MODELING_RF       <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name){
  cat("Model Type is Random Forest \n")
  
  train_y <- subset(train_data, select = yvar_name)
  train_x <- subset(train_data, select = xvar_name)
  
  # tune
  rf.fit <- rfcv(trainx = train_x, trainy = train_y[,1], cv.fold = 5, ntree = 500)
  rf.opt <- rf.fit$n.var[which.min(rf.fit$error.cv)]
  
  # Fit
  formula  <- paste0(yvar_name,' ~ ', paste(xvar_name, collapse = "+"))
  rf_model <- randomForest(as.formula(formula), data=train_data, mtry=rf.opt)
  
  # Pred
  train_data$pred  <- predict(rf_model, newdata = train_data)
  test_data$pred   <- predict(rf_model, newdata = test_data)
  
  # Result
  RESULT <- list('working_date' = working_date,
                 'model_type'  = 'RF',
                 'yvar'        = yvar_name,
                 'xvar'        = xvar_name,
                 'model'       = rf_model,
                 'train_res'   = train_data,
                 'test_res'    = test_data)
  
  # Save
  save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'RF.RData', sep = '/'))
  
  return(RESULT)
}


MODELING_XGBOOST  <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name){
  
  cat("Model Type is Xgboost \n")
  train_label   <- subset(train_data, select = yvar_name)
  train_feature <- subset(train_data, select = xvar_name)
  test_label    <- subset(test_data, select = yvar_name)
  test_feature  <- subset(test_data, select = xvar_name)
  train_d_mat <- xgb.DMatrix(data = as.matrix(train_feature), label = train_label[,1])
  test_d_mat  <- xgb.DMatrix(data = as.matrix(test_feature),  label = test_label[,1])
  
  # Tune
  xgb.grid.search <- expand.grid(max_depth = c(6), #default = 6
                                 eta = 0.3, #default = 0.3
                                 colsample_bytree = c(1), #default = 1
                                 subsample = c(1),
                                 alpha = c(1)) #default = 1
  perf.xgb.mat <- matrix(0, nrow(xgb.grid.search), 2)
  colnames(perf.xgb.mat) <- c("iter","score")
  
  # Grid Search
  for(i in 1:nrow(xgb.grid.search)){
    params.xgb <- list(objective = "reg:squarederror",
                       booster = "gbtree",
                       eta = xgb.grid.search[i, "eta"], #default = 0.3
                       max_depth = xgb.grid.search[i, "max_depth"], #default=6
                       subsample = xgb.grid.search[i, "subsample"],
                       colsample_bytree = xgb.grid.search[i, "colsample_bytree"],
                       alpha = xgb.grid.search[i, "alpha"])
    
    if (length(unique(train_label[,1])) > 1){
      # train target이 단일값이 아닌 경우
      xgbcv <- xgb.cv(params = params.xgb, data = train_d_mat, nrounds = 200, nfold = 5,
                      print_every_n = 10, early_stopping_rounds = 10, maximize = F, verbose = FALSE)
    } else {
      # train target이 단일값 인 경우(xgb.cv에서 fold를 나누는데 Error가 발생하여 folds를 지정)
      folds = split( 1:dim(train_label)[1] , sample(5, dim(train_label)[1]  , repl = TRUE) )
      xgbcv <- xgb.cv(params = params.xgb, data = train_d_mat, nrounds = 200, folds = folds,
                      print_every_n = 10, early_stopping_rounds = 10, maximize = F, verbose = FALSE)
    }
    perf.xgb.mat[i,] <- c(xgbcv$best_iteration, min(xgbcv$evaluation_log$test_rmse_mean))
  }
  
  # Find Best Tuning Parameters
  final.perf.xgb.mat <- cbind(xgb.grid.search,perf.xgb.mat)
  xgb.opt.par <- final.perf.xgb.mat[which.min(final.perf.xgb.mat[,"score"]),]
  
  # Fit
  params.opt.xgb <- list(objective = "reg:squarederror",booster = "gbtree",
                         eta = xgb.opt.par$eta, max_depth = xgb.opt.par$max_depth,
                         subsample = xgb.opt.par$subsample,colsample_bytree = xgb.opt.par$colsample_bytree,
                         alpha = xgb.grid.search[i,"alpha"])
  xgb_model <- xgb.train(data = train_d_mat, params=params.opt.xgb, nrounds = xgb.opt.par$iter, verbose = FALSE)
  
  # Pred
  train_data$pred  <- predict(xgb_model,train_d_mat)
  test_data$pred   <- predict(xgb_model,test_d_mat)
  
  # Result
  RESULT <- list('working_date' = working_date, 'model_type'   = 'XGBOOST',
                 'yvar'         = yvar_name   , 'xvar'         = xvar_name,
                 'model'        = xgb_model   , 'train_res'    = train_data,
                 'test_res'     = test_data)
  
  # Save
  save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'XGBOOST.RData', sep = '/'))
  return(RESULT)
}

MODELING_LIGHTGBM <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name){
  
  cat("Model Type is Lightgbm \n")
  train_label   <- subset(train_data, select = yvar_name)
  train_feature <- subset(train_data, select = xvar_name)
  test_label    <- subset(test_data, select = yvar_name)
  test_feature  <- subset(test_data, select = xvar_name)
  train_d_mat <- lgb.Dataset(data = as.matrix(train_feature), label = train_label[,1])
  test_d_mat  <- lgb.Dataset(data = as.matrix(test_feature) , label = test_label[,1])
  
  #tune grid
  lgb.grid.search <- expand.grid(num_leaves = c(3),       # default 31
                                 learning_rate = c(0.1),  # default 0.1
                                 feature_fraction = c(1), # default 1
                                 bagging_fraction = c(1), # default 0
                                 max_bin = c(255)         # dafault 255
  )
  perf.lgb.mat <- matrix(0,nrow(lgb.grid.search),2)
  colnames(perf.lgb.mat) <- c("iter","score")
  
  # Grid Search
  for(i in 1:nrow(perf.lgb.mat)){
    params.lgb <- list(objective        = "regression",
                       boosting         = "gbdt",
                       metric           = 'rmse',
                       num_leaves       = lgb.grid.search[i,"num_leaves"],
                       learning_rate    = lgb.grid.search[i,"learning_rate"],
                       feature_fraction = lgb.grid.search[i,"feature_fraction"],
                       bagging_fraction = lgb.grid.search[i,"bagging_fraction"],
                       max_bin          = lgb.grid.search[i,"max_bin"]
    )
    lgbcv <- lgb.cv(params = params.lgb, data = train_d_mat, nrounds = 200, nfold = 5,early_stopping_rounds = 10, verbose = -1)
    perf.lgb.mat[i, ] <- c(lgbcv$best_iter, lgbcv$best_score)
  }
  
  # Find Best Tuning Parameters
  final.perf.lgb.mat <- cbind(lgb.grid.search,perf.lgb.mat)
  lgb.opt.par <- final.perf.lgb.mat[which.min(final.perf.lgb.mat[,"score"]), ]
  
  # Fit
  params.opt.lgb <- list(objective        = "regression",boosting         = "gbdt",metric           = 'rmse',
                         num_leaves       = lgb.opt.par$num_leaves,learning_rate    = lgb.opt.par$learning_rate,
                         feature_fraction = lgb.opt.par$feature_fraction,bagging_fraction = lgb.opt.par$bagging_fraction,
                         max_bin          = lgb.opt.par$max_bin)
  
  lgb_model <- lgb.train(data = train_d_mat, params = params.opt.lgb,nrounds = lgb.opt.par$iter, verbose = -1)
  
  # Pred
  train_data$pred <- predict(lgb_model, as.matrix(train_feature))
  test_data$pred  <- predict(lgb_model, as.matrix(test_feature))
  
  # Result
  RESULT <- list('working_date' = working_date,'model_type'   = 'LIGHTGBM',
                 'yvar'         = yvar_name   ,'xvar'         = xvar_name,
                 'model'        = paste(model_path, working_date, yvar_name, 'MODEL','LIGHTGBM.LGB', sep = '/'),
                 'train_res'    = train_data  ,'test_res'     = test_data)
  # Save
  save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'LIGHTGBM.RData', sep = '/'))
  lgb.save(lgb_model,filename = paste(model_path, working_date, yvar_name, 'MODEL', 'LIGHTGBM.LGB',sep = '/'))
  return(RESULT)
}

MODELING_ANN      <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name){
  cat("Model Type is ANN \n")
  
  nn_train_x <- train_data[,xvar_name]
  nn_test_x  <- test_data[,xvar_name]
  nn_train_y <- train_data[,yvar_name]
  nn_test_y  <- test_data[,yvar_name]
  
  # scale info
  max_val <- apply(nn_train_x, 2, max)
  min_val <- apply(nn_train_x, 2, min)
  scale_info <- data.frame(max = max_val, min = min_val)
  nn_train_x <- nn_train_x[,rownames(scale_info)]
  nn_test_x  <- nn_test_x[,rownames(scale_info)]
  
  # min max scale
  ann_minmax_scale <- function(data, scale_info){
    max <- scale_info['max'][,1] ; min <- scale_info['min'][,1]
    ret <- apply(data,1, function(x){(x-min)/(max - min + 1e-7)})
    return(t(ret))
  }
  nn_train_x_scale <- ann_minmax_scale(nn_train_x, scale_info)
  nn_test_x_scale  <- ann_minmax_scale(nn_test_x, scale_info)
  
  # Fit
  model <- keras_model_sequential()%>%
    layer_dense(units = 32,activation='relu', input_shape = c(dim(nn_train_x)[2])) %>%
    layer_dense(units = 8 ,activation='relu') %>%
    layer_dense(units = 1, activation = 'linear')
  
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate=0.05),
    loss = 'mse'
  )
  
  history <- model %>% fit(
    x = nn_train_x_scale,
    y = nn_train_y,
    batch_size = max(round(nrow(nn_train_x_scale)/5, 0), 32),
    epoch = 500,
    validation_split = 0.2,
    shuffle = T,
    view_metrics = F,
    verbose = 0,
    callbacks = list(
      # callback_csv_logger(SaveModelLOGNM),
      callback_early_stopping(monitor = "val_loss", patience = 50, restore_best_weights = T,verbose = 0),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.8, patience = 10, verbose = 0, mode = "auto")
    )
  )
  
  # Pred
  train_data$pred <- model %>% predict(nn_train_x_scale)
  test_data$pred  <- model %>% predict(nn_test_x_scale)
  
  # Result
  RESULT <- list('working_date' = working_date, 'model_type' = 'ANN',
                 'yvar'       = yvar_name     , 'xvar'       = xvar_name,
                 'model'      = paste(model_path, working_date, yvar_name, 'MODEL', 'ANN.h5',sep = '/'),
                 'train_res'  = train_data    , 'test_res'   = test_data,
                 'scale_info' = scale_info)
  
  # Save
  save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'ANN.RData', sep = '/'))
  save_model_hdf5(model, filepath = paste(model_path, working_date, yvar_name, 'MODEL', 'ANN.h5',sep = '/'))
  return(RESULT)
}

MODELING_LSTM     <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name, only.y = F){
  cat(paste0("Model Type is ",ifelse(only.y == F, 'LSTM', 'LSTM_Y')),'\n')
  
  # only.y == T : y변수의 과거 값만을 사용하여 모델 생성
  if(only.y == T){xvar_name <- xvar_name[grepl(stringr::str_replace(yvar_name,"y_",""),xvar_name)]}
  
  test_start_time <- min(test_data[, 'predict_time'])
  lstm_full_data <- rbind(train_data, test_data)
  lstm_full_x  <- lstm_full_data[, xvar_name]
  lstm_full_y  <- lstm_full_data[, yvar_name]
  
  # TimeStep 설정
  nTimeStep <- 5

  # Batch Dimension
  dim_batch <- dim(lstm_full_x)[1] - 1 * (nTimeStep-1)
  idx_array <- sapply(1:dim_batch, function(s) {
    seq(from = s, to = 1 * (nTimeStep - 1) + s, by = 1)
  })
  
  # Lstm Data 생성
  lstm_array_x <- array(NA,dim = c(dim_batch, nTimeStep, dim(lstm_full_x)[2]))
  lstm_array_y <- array(NA,dim = c(dim_batch, nTimeStep, 1))
  for(idx in 1:dim_batch){
    lstm_array_x[idx,,] <- as.matrix(lstm_full_x[idx_array[, idx],])
    lstm_array_y[idx,,] <- lstm_full_y[idx_array[, idx]]
  }
  
  # t시점의 predcit time 가져오기(ex : nTimesStep이 5일 때, t-4, t-3, t-2, t-1, t)
  date_vec <- lstm_full_data[,'predict_time'][idx_array[nTimeStep,]]
  
  # 학습, 테스트 데이터로 나누기
  lstm_array_train_x <- lstm_array_x[(date_vec < test_start_time),,]
  lstm_array_train_y <- lstm_array_y[(date_vec < test_start_time),,]
  lstm_array_test_x  <- lstm_array_x[(date_vec >= test_start_time),,]
  lstm_array_test_y  <- lstm_array_y[(date_vec >= test_start_time),,]
  
  # Scaling
  min_val <- apply(lstm_array_train_x, 3, min)
  max_val <- apply(lstm_array_train_x, 3, max)
  scale_info <- data.frame(max = max_val, min = min_val)
  rownames(scale_info) <- xvar_name
  
  lstm_minmax_scale <- function(data, scale_info){
    max <- scale_info['max'][,1]
    min <- scale_info['min'][,1]
    eps <- 1e-7
    for(idx in 1:dim(data)[1]){
      data[idx,,] <- t(apply(data[idx,,], 1 ,function(x){(x - min)/(max - min + eps)}))
    }
    return(data)
  }
  
  # minmax scaling
  lstm_array_train_x <- lstm_minmax_scale(lstm_array_train_x,scale_info)
  lstm_array_test_x  <- lstm_minmax_scale(lstm_array_test_x,scale_info)
  
  # Fit
  model <- keras_model_sequential() %>% 
    layer_lstm(units = 128, input_shape = c(nTimeStep, dim(lstm_full_x)[2]), return_sequences = TRUE) %>% 
    layer_dense(1, activation = 'linear')
  
  model %>% compile(
    optimizer = optimizer_rmsprop(learning_rate=0.05),
    loss = 'mse'
  )
  
  history <- model %>% fit(
    x = lstm_array_train_x,
    y = lstm_array_y,
    batch_size = max(round(dim(lstm_array_train_x)[1]/5, 0), 32),
    epoch = 500,
    validation_split = 0.2,
    shuffle = T,
    view_metrics = F,
    verbose = 0,
    callbacks = list(
      # callback_csv_logger(SaveModelLOGNM),
      callback_early_stopping(monitor = "val_loss", patience = 50, restore_best_weights = T,
                              verbose = 0),
      callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.8, patience = 10, verbose = 0,
                                    mode = "auto")
    )
  )
  
  # Pred
  train_pred      <- model %>% predict(lstm_array_train_x)
  train_data$pred <- c(rep(NA, nTimeStep - 1), train_pred[, dim(train_pred)[2], ])
  test_pred       <- model %>% predict(lstm_array_test_x)
  test_data$pred  <- test_pred[, dim(test_pred)[2],]
  
  # Result
  RESULT <- list('working_date' = working_date,
                 'model_type'   = ifelse(only.y == F, 'LSTM', 'LSTM_Y'),
                 'yvar'         = yvar_name,
                 'xvar'         = xvar_name,
                 'model'        = paste(model_path, working_date, yvar_name, 'MODEL',ifelse(only.y == T,'LSTM_Y.h5','LSTM.h5'), sep = '/'),
                 'train_res'    = train_data,
                 'test_res'     = test_data,
                 'scale_info'   = scale_info,
                 'train_array'  = lstm_array_train_x,
                 'test_array'   = lstm_array_test_x)
  
  # Save
  if(only.y == T){
    save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'LSTM_Y.RData', sep = '/'))
    save_model_hdf5(model,filepath = paste(model_path, working_date, yvar_name, 'MODEL', 'LSTM_Y.h5',sep = '/'))
  } else {
    save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'LSTM.RData', sep = '/'))
    save_model_hdf5(model,filepath = paste(model_path, working_date, yvar_name, 'MODEL', 'LSTM.h5',sep = '/'))
  }
  
  return(RESULT)
}

MODELING_ARIMAX   <- function(working_date, model_path, train_data, test_data, yvar_name, xvar_name, only.y = F){
  cat(paste0("Model Type is ",ifelse(only.y == F, 'ARIMAX', 'ARIMA_Y')),'\n')
  
  # only.y == T : y변수의 과거 값만을 사용하여 모델 생성
  if(only.y == T){xvar_name <- xvar_name[grepl(stringr::str_replace(yvar_name,"y_",""),xvar_name)]}
  
  # Train data
  train_arima_y <- train_data[, yvar_name]
  train_arima_x <- train_data[, xvar_name]
  
  # unique value = 1인 변수 제외
  excluded_xnames <- names(which(apply(train_arima_x, 2, function(x){length(unique(x))}) == 1))
  train_arima_x <- train_arima_x[,!(xvar_name %in% excluded_xnames)]
  
  # corr != 1 변수
  corr_mat <- cor(train_arima_x)
  corr_mat[lower.tri(corr_mat, diag = TRUE)] <- NA                              # put NA
  corr_mat <- as.data.frame(as.table(corr_mat), stringsAsFactors = FALSE)       # as a dataframe
  corr_mat <- na.omit(corr_mat)                                                 # remove NA
  corr_mat <- corr_mat[with(corr_mat, order(-Freq)), ]                          # order by correlation
  
  # Final train data
  excluded_xnames <- c(excluded_xnames, unique(unlist(corr_mat[abs(corr_mat[, 3]) == 1, 1:2])))
  train_arima_x <- as.matrix(train_arima_x[, !(colnames(train_arima_x) %in% excluded_xnames)])
  
  # Test data
  test_arima_y <- test_data[, yvar_name]
  test_arima_x <- test_data[, xvar_name]
  test_arima_x <- test_arima_x[, !(colnames(test_arima_x) %in% excluded_xnames)]
  xvar_name    <- colnames(test_arima_x)
  
  # Fit
  arimax <- tryCatch(forecast::auto.arima(y = train_arima_y, xreg = train_arima_x),error = function(e){"err"})
  if(!is.character(arimax)){
    if( mean("intercept" %in% names(arimax$coef)) == 1 ){
      arimax <- auto.arima(y = train_arima_y)
      train_data$pred <- forecast::forecast(arimax, h = dim(train_arima_x)[1])$mean
      test_data$pred  <- forecast::forecast(arimax, h = dim(test_arima_x)[1])$mean
      xvar_name       <- NULL
    } else {
      train_data$pred <- forecast::forecast(arimax, xreg = as.matrix(train_arima_x))$mean
      test_data$pred  <- forecast::forecast(arimax, xreg = as.matrix(test_arima_x))$mean
    }
  } else {
    arimax <- auto.arima(y = train_arima_y)
    train_data$pred <- forecast::forecast(arimax, h = dim(train_arima_x)[1])$mean
    test_data$pred  <- forecast::forecast(arimax, h = dim(test_arima_x)[1])$mean
    xvar_name       <- NULL
  }

  # Result
  RESULT <- list('working_date'    = working_date,'model_type'      = ifelse(only.y == T, 'ARIMAX_Y','ARIMAX'),
                 'yvar'            = yvar_name   ,'xvar'            = xvar_name,
                 'model'           = arimax      ,'train_res'       = train_data,
                 'test_res'        = test_data   ,'excluded_xnames' = excluded_xnames)
  
  # Save
  if(only.y == T){
    save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'ARIMAX_Y.RData', sep = '/'))
  } else {
    save(RESULT,file = paste(model_path, working_date, yvar_name, 'MODEL', 'ARIMAX.RData', sep = '/'))
  }
  
  return(RESULT)
}

#-- 함수 2. Evaluate Function
# result_list : 방법론 별 MODELING 함수 결과 모음 LIST
# thrs        : Accuracy 계산을 위한 Threshold  
# type        : mape 또는 acc (ex : if type = 'mape', then 'mape' 순으로 랭킹/'mape' 동차 시 acc 순으로 랭킹)
EVAL_FN <- function(result_list, thrs, type){
  
  cat("Eval Type :", toupper(type),"\n")
  
  # mape
  mape_fn <- function(y_true, y_pred){
    mape_val <- abs((y_true - y_pred)/(y_true + 1e-7))
    mape_val <- ifelse(mape_val >= 1, 1, mape_val)
    return(mean(mape_val, na.rm = T))
  }
  
  # acc
  acc_fn <- function(y_true, y_pred, thrs){
    y_true <- (y_true >= thrs) + 0
    y_pred <- (y_pred >= thrs) + 0
    
    tp <- sum((y_true == 1) * (y_pred == 1), na.rm = T)
    tn <- sum((y_true == 0) * (y_pred == 0), na.rm = T)
    fp <- sum((y_true == 0) * (y_pred == 1), na.rm = T)
    fn <- sum((y_true == 1) * (y_pred == 0), na.rm = T)
    
    # return acc
    return((tp + tn) / (tp + tn + fp + fn))
  }
  
  # EVAL
  EVAL <- list()
  for(i in 1:length(result_list)){
    TEMP       <- result_list[[i]]
    yvar       <- TEMP$yvar
    model      <- TEMP$model_type
    train_mape <- mape_fn(TEMP$train_res[,yvar_name], TEMP$train_res[,'pred'])
    test_mape  <- mape_fn(TEMP$test_res[,yvar_name], TEMP$test_res[,'pred'])
    train_acc  <- acc_fn(TEMP$train_res[,yvar_name], TEMP$train_res[,'pred'], thrs)
    test_acc   <- acc_fn(TEMP$test_res[,yvar_name], TEMP$test_res[,'pred'], thrs)
    
    # return
    ret <- data.frame(model = model, yvar = yvar, acc_thrs = thrs ,type = type,
                      train_mape = train_mape , train_acc = train_acc,
                      test_mape = test_mape, test_acc = test_acc)
    EVAL[[i]] <- ret
    
  }
  EVAL <- setDF(rbindlist(EVAL))
  
  # Sort
  if(tolower(type) == "mape"){
    EVAL <- EVAL[order(EVAL[,'test_mape'], -EVAL[,'test_acc']),]
    
  } else if(tolower(type) == 'acc'){
    EVAL <- EVAL[order(-EVAL[,'test_acc'], EVAL[,'test_mape']),]
  }
  
  # Ranking
  EVAL$ranking <- 1:dim(EVAL)[1]
  return(EVAL)
}

#-- 함수 3. Model Save Function
# working_date : 학습 일자
# model_path   : 모델 저장 위치
# yvar_name    : y변수 명
# eval_df      : EVAL_FN 결과
SAVE_FN <- function(working_date, model_path,yvar_name, eval_df){
  
  best_model_nm <- eval_df[eval_df['ranking'] == 1,'model']
  type <- toupper(unique(eval_df$type))
  yvar_name <- unique(eval_df$yvar)
  

  if (best_model_nm %in% c('LSTM','LSTM_Y','ANN')){
    ori_path_rds = paste(model_path,working_date,yvar_name,'MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    new_path_rds = paste(model_path,working_date,yvar_name,'BEST_MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    ori_path_h5 = paste(model_path,working_date,yvar_name,'MODEL',paste0(best_model_nm,'.h5'),sep = '/')
    new_path_h5 = paste(model_path,working_date,yvar_name,'BEST_MODEL',paste0(best_model_nm,'.h5'),sep = '/')
    
    file.copy(ori_path_rds, new_path_rds)
    file.copy(ori_path_h5, new_path_h5)
  } else if (best_model_nm == 'LIGHTGBM') {
    ori_path_rds = paste(model_path,working_date,yvar_name,'MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    new_path_rds = paste(model_path,working_date,yvar_name,'BEST_MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    ori_path_mdl = paste(model_path,working_date,yvar_name,'MODEL',paste0(best_model_nm,'.LGB'),sep = '/')
    new_path_mdl = paste(model_path,working_date,yvar_name,'BEST_MODEL',paste0(best_model_nm,'.LGB'),sep = '/')
    
    file.copy(ori_path_rds, new_path_rds)
    file.copy(ori_path_mdl, new_path_mdl)
  } else {
    ori_path = paste(model_path,working_date,yvar_name,'MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    new_path = paste(model_path,working_date,yvar_name,'BEST_MODEL',paste0(best_model_nm,'.RData'),sep = '/')
    file.copy(ori_path, new_path)
  }
  
  # save
  cat('[ Working Date :', working_date,'][Best Model Save :',best_model_nm,'(',type,')]','\n')
}
