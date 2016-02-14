##########################################################################
# Regressive XGBoost with optimal cuts
# LB score ~ 0.65
##########################################################################
library(caret)
library(reshape2)
library(dplyr)
library(xgboost)
library(dummies)
library(Metrics)
library(mlr)

setwd('~/leave_academia/kaggle/prudential')
load('preprocessed.RData')

##########################################################################
##########################################################################
### Customized objectives
##########################################################################
##########################################################################
# SQWKfun with customized cuts
SQWKfun = function(x = seq(1.5, 7.5, by = 1), preds, true_vals) {
  cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
  preds = as.numeric(Hmisc::cut2(preds, cuts))
  kappa = Metrics::ScoreQuadraticWeightedKappa(preds, true_vals, 1, 8)
  return(kappa) #because the function minimizes
}

# for mlr
mySQWKfun <- function(task, model, pred, feats, m) { # already with optimal cuts
    SQWKfun(preds = pred$data$response, true_vals = pred$data$truth)
}
my.sqwk <- makeMeasure(id = 'my.sqwk', minimize = F,
                       properties = c('regr', 'response'), fun = mySQWKfun, best = 1, worst = 0)

# to create predictions
cut.pred <- function(x, cutpar) {
    cutpar <- sort(cutpar) # some optimizations are not sorted
    cuts <- c(min(x), cutpar[1], cutpar[2], cutpar[3], cutpar[4],
              cutpar[5], cutpar[6], cutpar[7], max(x))
    preds <- as.numeric(Hmisc::cut2(x, cuts))
    preds
}


# Separate test-train

indx.test <- is.na(all.data$response)
train.data <- all.data[!indx.test, -1]
test.data <- all.data[indx.test, -2]

# train-test split
set.seed(12)
split <- createDataPartition(train.data$response, p = 0.7)[[1]]
tr.data <- train.data[split,]
ts.data <- train.data[-split, ]

##########################################################################
##########################################################################
# Feature Selection by Filter: did not help much but record the script
# information.gain: select all features ~ 0.64
# rank.correlation: select all features ~ 0.64
# gain.ratio: select all features ~ 0.64
# chi.squared: select all features ~ 0.64
##########################################################################
##########################################################################

library(mlr)
# Step1: tasks
train.task <- makeRegrTask(data = tr.data, target = 'response')
test.task <- makeRegrTask(data = ts.data, target = 'response')

set.seed(954)
lrn <- makeFilterWrapper(learner = 'regr.xgboost', fw.method = 'chi.squared')
lrn$next.learner$par.vals <- list( # parameters not for tuning
  #nthread             = 30,
  eta = 0.0719,
  gamma = 0.451,
  nrounds             = 150,
  #print.every.n       = 5,
  objective           = "reg:linear",
  max_depth = 10,
  colsample_bytree = 0.623,
  #min_child_weight = 3,
  eval_metric = xgb.objective.sqwk,
  subsample = 0.991
)

# Step3: resampling and performace metrics
rdesc <- makeResampleDesc('CV', iters = 5)
ps <- makeParamSet(makeDiscreteParam('fw.perc', values = c(0.40, 0.5, 0.60, 0.80, 1)))
 

#Step4 : tuning
res <- tuneParams(lrn, task = train.task, resampling = rdesc, par.set = ps,
                  control = makeTuneControlGrid(), measures = list(my.sqwk, mse))

##########################################################################
##########################################################################
### MLR tuning
##########################################################################
##########################################################################

library(mlr)
# Step1: tasks
train.task <- makeRegrTask(data = tr.data, target = 'response')
test.task <- makeRegrTask(data = ts.data, target = 'response')

# Step2: learner and parameters
set.seed(954)
lrn <- makeLearner('regr.xgboost')
lrn$par.vals <- list( # parameters not for tuning
  #nthread             = 30,
  nrounds             = 150,
  #print.every.n       = 5,
  objective           = "reg:linear"
  #depth = 20,
  #colsample_bytree = 0.66,
  #min_child_weight = 3,
  #subsample = 0.71
)

ps.set <- makeParamSet(
    makeNumericParam('eta', lower = 0.05, upper = 0.2),
    makeNumericParam('gamma', lower = 0, upper = 1),
    makeNumericParam('subsample', lower = 0.5, upper = 1),
    makeNumericParam('colsample_bytree', lower = 0.5, upper = 1),
    makeDiscreteParam('max_depth', values = c(10, 15, 20))
)

# Step3: search and validation 
ctrl <- makeTuneControlRandom(budget = 20, maxit = 20)
rdesc <- makeResampleDesc('CV', iters = 5L)

# Step4: tune
tune.lrn <- tuneParams(lrn, task = train.task, resampling = rdesc, par.set = ps.set, control = ctrl,
                       measures = my.sqwk)
lrn <- setHyperPars(lrn, par.vals = tune.lrn$x)

# Step5: predict
train.lrn <- train(lrn, train.task)
test.pred <- predict(train.lrn, test.task)
mySQWKfun(pred = test.pred) # 0.5921376

# Step6: optimal cuts via cross validation
train.task <- makeRegrTask(data = train.data, target = 'response')
validated.lrn <- crossval(lrn, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)

to_optimize <-  function(x) {
    -SQWKfun(x, validated.lrn$pred$data$response, as.numeric(validated.lrn$pred$data$truth))
}

optCuts = optim(seq(1.5, 7.5, by = 1), to_optimize)
optpars <- optCuts$par
SQWKfun(optCuts$par, test.pred$data$response, test.pred$data$truth) # = 0.6424829

##########################################################################
##########################################################################
## Predict
##########################################################################
##########################################################################

# Step1: learner and parameters
set.seed(954)
lrn <- makeLearner('regr.xgboost')
lrn$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200 # found that on full train a bit more rounds help
)

global.train.task <-  makeRegrTask(data = train.data, target = 'response')
global.train <- train(lrn, global.train.task)

global.pred <- predict(global.train, newdata = test.data[,-1]) # newdata: no labels for test.data
out <- data.frame(Id = test.data$id, Response = sapply(global.pred$data$response, cut.pred, optpars))
write.table(file = '2016feb2_xgboost_imputed.csv', out, quote = F, row.names = F, sep = ',') # = 0.65583
	
