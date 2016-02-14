##########################################################################
##########################################################################
# Ensembling predictors via voting
# Improves LB ~ 0.66
##########################################################################
##########################################################################
# Package ExtraTree uses Java: 12Gb needed in the Heap!
options(java.parameters = "-Xmx12g") # to set before loading mlr !!
library(caret)
library(reshape2)
library(dplyr)
library(xgboost)
library(dummies)
library(Metrics)
library(mlr)

setwd('~/leave_academia/kaggle/prudential')
load('preprocessed.RData')

# Separate test-train
indx.test <- is.na(all.data$response)
train.data <- all.data[!indx.test, -1]
test.data <- all.data[indx.test, -2]

##########################################################################
##########################################################################
### Evaluation metrics: consider putting this in an extra help file
##########################################################################
##########################################################################

# Compute best prediction subordinate to cuts
SQWKfun = function(x = seq(1.5, 7.5, by = 1), preds, true_vals) {
  x <- sort(x)
  cuts = c(min(preds), x[1], x[2], x[3], x[4], x[5], x[6], x[7], max(preds))
  preds = as.numeric(Hmisc::cut2(preds, cuts))
  kappa = Metrics::ScoreQuadraticWeightedKappa(preds, true_vals, 1, 8)
  return(kappa) #because the function minimizes
}

mySQWKfun <- function(task, model, pred, feats, m) {
    SQWKfun(preds = pred$data$response, true_vals = pred$data$truth)
}

my.sqwk <- makeMeasure(id = 'my.sqwk', minimize = F, properties = c('regr', 'response'), fun = mySQWKfun, best = 1, worst = 0)
xgb.objective.sqwk <- function(pred, dtrain) {
    val = SQWKfun(preds = pred, true_vals = getinfo(dtrain, 'label'))
    return(list(metric = 'my.sqwk', value = val))
}

# train task
train.task <- makeRegrTask(data = train.data, target = 'response')


##########################################################################
##########################################################################
## Create 8 XGBoost models for ensembling
##########################################################################
##########################################################################
set.seed(234)
lrn.xgb1 <- makeLearner('regr.xgboost')
lrn.xgb1$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200   
)

set.seed(152)
validated.xgb1 <- crossval(lrn.xgb1, train.task, iter = 5, measures = my.sqwk, show.info = TRUE) # to find optimal cuts
save(validated.xgb1, file = 'stacked/validated-xgb1.RData')
set.seed(50)
trained.xgb1 <- train(lrn.xgb1, train.task)
predicted.xgb1 <- predict(trained.xgb1, newdata = test.data[, -1])
save(predicted.xgb1, file = 'stacked/predicted-xgb1.RData')

set.seed(785)
lrn.xgb2 <- makeLearner('regr.xgboost')
lrn.xgb2$par.vals <- list( 
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(178)
validated.xgb2 <- crossval(lrn.xgb2, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb2, file = 'stacked/validated-xgb2.RData')
set.seed(181)
trained.xgb2 <- train(lrn.xgb2, train.task)
predicted.xgb2 <- predict(trained.xgb2, newdata = test.data[, -1])
save(predicted.xgb2, file = 'stacked/predicted-xgb2.RData')


set.seed(519)
lrn.xgb3 <- makeLearner('regr.xgboost')
lrn.xgb3$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(890)
validated.xgb3 <- crossval(lrn.xgb3, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb3, file = 'stacked/validated-xgb3.RData')
set.seed(41)
trained.xgb3 <- train(lrn.xgb3, train.task)
predicted.xgb3 <- predict(trained.xgb3, newdata = test.data[, -1])
save(predicted.xgb3, file = 'stacked/predicted-xgb3.RData')


set.seed(683)
lrn.xgb4 <- makeLearner('regr.xgboost')
lrn.xgb4$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(566)
validated.xgb4 <- crossval(lrn.xgb4, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb4, file = 'stacked/validated-xgb4.RData')
set.seed(450)
trained.xgb4 <- train(lrn.xgb4, train.task)
predicted.xgb4 <- predict(trained.xgb4, newdata = test.data[, -1])
save(predicted.xgb4, file = 'stacked/predicted-xgb4.RData')

set.seed(43)
lrn.xgb5 <- makeLearner('regr.xgboost')
lrn.xgb5$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(67)
validated.xgb5 <- crossval(lrn.xgb5, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb5, file = 'stacked/validated-xgb5.RData')
set.seed(2)
trained.xgb5 <- train(lrn.xgb5, train.task)
predicted.xgb5 <- predict(trained.xgb5, newdata = test.data[, -1])
save(predicted.xgb5, file = 'stacked/predicted-xgb5.RData')

set.seed(32)
lrn.xgb6 <- makeLearner('regr.xgboost')
lrn.xgb6$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(317)
validated.xgb6 <- crossval(lrn.xgb6, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb6, file = 'stacked/validated-xgb6.RData')
set.seed(997)
trained.xgb6 <- train(lrn.xgb6, train.task)
predicted.xgb6 <- predict(trained.xgb6, newdata = test.data[, -1])
save(predicted.xgb6, file = 'stacked/predicted-xgb6.RData')

set.seed(843)
lrn.xgb7 <- makeLearner('regr.xgboost')
lrn.xgb7$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(51)
validated.xgb7 <- crossval(lrn.xgb7, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb7, file = 'stacked/validated-xgb7.RData')
set.seed(86)
trained.xgb7 <- train(lrn.xgb7, train.task)
predicted.xgb7 <- predict(trained.xgb7, newdata = test.data[, -1])
save(predicted.xgb7, file = 'stacked/predicted-xgb7.RData')

set.seed(9)
lrn.xgb8 <- makeLearner('regr.xgboost')
lrn.xgb8$par.vals <- list(
    eta = 0.1,
    gamma = 0.5,
    subsample = 1,
    colsample_bytree = 0.6,
    max_depth = 10,
    nrounds = 200  
)

set.seed(83)
validated.xgb8 <- crossval(lrn.xgb8, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xgb8, file = 'stacked/validated-xgb8.RData')
set.seed(4)
trained.xgb8 <- train(lrn.xgb8, train.task)
predicted.xgb8 <- predict(trained.xgb8, newdata = test.data[, -1])
save(predicted.xgb8, file = 'stacked/predicted-xgb8.RData')


# rm(trained.xgb1, trained.xgb2, trained.xgb3, trained.xgb4)
# system('xclock') # signals end of fitting

##########################################################################
### Random Forests
##########################################################################
set.seed(368)
lrn.rf1 <- makeLearner('regr.randomForest')
lrn.rf1$par.vals = list(
    ntree = 500,
    nodesize = 5
)

set.seed(172)
validated.rf1 <- crossval(lrn.rf1, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.rf1, file = 'stacked/validated-rf1.RData')
set.seed(441)
trained.rf1 <- train(lrn.rf1, train.task)
predicted.rf1 <- predict(trained.rf1, newdata = test.data[, -1])
save(predicted.rf1, file = 'stacked/predicted-rf1.RData')
rm(trained.rf1) # random forest takes a lot of memory

set.seed(368)
lrn.rf2 <- makeLearner('regr.randomForest')
lrn.rf2$par.vals = list(
    ntree = 500,
    nodesize = 5
)

set.seed(674)
validated.rf2 <- crossval(lrn.rf2, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.rf2, file = 'stacked/validated-rf2.RData')
set.seed(468)
trained.rf2 <- train(lrn.rf2, train.task)
predicted.rf2 <- predict(trained.rf2, newdata = test.data[, -1])
save(predicted.rf2, file = 'stacked/predicted-rf2.RData')
rm(trained.rf2)
system('xclock') # end of fitting ~ 5hrs

##########################################################################
### Extra Trees
##########################################################################
set.seed(132)
lrn.xtree1 <- makeLearner('regr.extraTrees')
lrn.xtree1$par.val <- list(
    ntree = 500,
    mtry = 35,
    nodesize = 20,
    numRandomCuts = 2,
    evenCuts = TRUE
)

set.seed(586)
validated.xtree1 <- crossval(lrn.xtree1, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xtree1, file = 'stacked/validated-xtr1.RData')
set.seed(368)
trained.xtree1 <- train(lrn.xtree1, train.task)
predicted.xtree1 <- predict(trained.xtree1, newdata = test.data[, -1])
save(predicted.xtree1, file = 'stacked/predicted-xtr1.RData')
rm(trained.xtree1) # takes a lot of memory

set.seed(648)
lrn.xtree2 <- makeLearner('regr.extraTrees')
lrn.xtree2$par.val <- list(
    ntree = 500,
    mtry = 35,
    nodesize = 30,
    numRandomCuts = 2,
    evenCuts = TRUE
)

set.seed(641)
validated.xtree2 <- crossval(lrn.xtree2, train.task, iter = 5, measures = my.sqwk, show.info = TRUE)
save(validated.xtree2, file = 'stacked/validated-xtr2.RData')
set.seed(293)
trained.xtree2 <- train(lrn.xtree2, train.task)
predicted.xtree2 <- predict(trained.xtree2, newdata = test.data[, -1])
save(predicted.xtree2, file = 'stacked/predicted-xtr2.RData')
rm(trained.xtree2) # fits in about 3hrs

##########################################################################
### Load models
##########################################################################
file.nms <- c('validated-xgb1.RData', 'validated-xgb2.RData',
              'validated-xgb3.RData', 'validated-xgb4.RData',
              'validated-xgb5.RData', 'validated-xgb6.RData',
              'validated-xgb7.RData', 'validated-xgb8.RData',
              'validated-rf1.RData', 'validated-rf2.RData',
              'validated-xtr1.RData', 'validated-xtr2.RData',
              'predicted-xgb1.RData', 'predicted-xgb2.RData',
              'predicted-xgb3.RData', 'predicted-xgb4.RData',
              'predicted-xgb5.RData', 'predicted-xgb6.RData',
              'predicted-xgb7.RData', 'predicted-xgb8.RData',
              'predicted-rf1.RData', 'predicted-rf2.RData',
              'predicted-xtr1.RData', 'predicted-xtr2.RData')

for(fnm in file.nms) {
    load(paste('stacked', fnm, sep='/'))
}

# stack validated 
id.vec <- validated.xgb1$pred$data$id
validated.stacked <- data.frame(id = id.vec, response = validated.xgb1$pred$data$truth,
                                xgb1 = validated.xgb1$pred$data$response,
                                xgb2 = validated.xgb2$pred$data[match(id.vec, validated.xgb2$pred$data$id),
                                                                'response'],
                                xgb3 = validated.xgb3$pred$data[match(id.vec, validated.xgb3$pred$data$id),
                                                                'response'],
                                xgb4 = validated.xgb4$pred$data[match(id.vec, validated.xgb4$pred$data$id),
                                                                'response'],
                                xgb5 = validated.xgb5$pred$data[match(id.vec, validated.xgb5$pred$data$id),
                                                                'response'],
                                xgb6 = validated.xgb6$pred$data[match(id.vec, validated.xgb6$pred$data$id),
                                                                'response'],
                                xgb7 = validated.xgb7$pred$data[match(id.vec, validated.xgb7$pred$data$id),
                                                                'response'],
                                xgb8 = validated.xgb8$pred$data[match(id.vec, validated.xgb8$pred$data$id),
                                                                'response'],
                                rf1 = validated.rf1$pred$data[match(id.vec, validated.rf1$pred$data$id),
                                                              'response'],
                                rf2 = validated.rf2$pred$data[match(id.vec, validated.rf2$pred$data$id),
                                                              'response'],
                                xtr1 = validated.xtree1$pred$data[match(id.vec, validated.xtree1$pred$data$id),
                                                              'response'],
                                xtr2 = validated.xtree2$pred$data[match(id.vec, validated.xtree2$pred$data$id),
                                                              'response'])

predicted.stacked <- data.frame(id = test.data$id, xgb1 = predicted.xgb1$data$response,
                                xgb2 = predicted.xgb2$data$response,
                                xgb3 = predicted.xgb3$data$response,
                                xgb4 = predicted.xgb4$data$response,
                                xgb5 = predicted.xgb5$data$response,
                                xgb6 = predicted.xgb6$data$response,
                                xgb7 = predicted.xgb7$data$response,
                                xgb8 = predicted.xgb8$data$response,
                                rf1 = predicted.rf1$data$response,
                                rf2 = predicted.rf2$data$response,
                                xtr1 = predicted.xtree1$data$response,
                                xtr2 = predicted.xtree2$data$response)

# to create predictions
cut.pred <- function(x, cutpar) {
    cutpar <- sort(cutpar) # some optimizations are not sorted
    cuts <- c(min(x), cutpar[1], cutpar[2], cutpar[3], cutpar[4],
              cutpar[5], cutpar[6], cutpar[7], max(x))
    preds <- as.numeric(Hmisc::cut2(x, cuts))
    preds
}

# Find optimal parameters for each individual model
optim.pars <- list()
for(i in 3:14) {
    print(paste('Optimize', i))
     to_optimize <-  function(x) {
         -SQWKfun(x, validated.stacked[,i], as.numeric(validated.stacked$response))
     }
    optim.pars[[colnames(validated.stacked)[i]]] <- optim(seq(1.5, 7.5, by = 1), to_optimize)
}

save(optim.pars, file = 'stacked/optim_pars.RData')

# generate predictions for all models
for(i in 2:13) {
    col_name <- colnames(predicted.stacked)[i]
    file_name <- paste('stacked/pred-', col_name, '.csv', sep = '')
    cat(paste('Generating file', file_name),'\n')
    out <- data.frame(Id = predicted.stacked$id, Response = sapply(predicted.stacked[col_name],
                                                                   cut.pred, optim.pars[[col_name]][['par']]))
    write.table(file = file_name, out, quote = F, row.names = F, sep = ',')
}

# load the data and create a data-frame for voting
fnames <- paste('stacked/pred-', colnames(predicted.stacked)[-1], '.csv', sep = '')
dflist <- list()
for(fn in fnames) {
    dflist[[fn]] <- read.csv(file = fn, header = T, stringsAsFactors = F)
}

dfpred <- dflist[[1]]
for(i in c(2:length(dflist))) {
    dfpred <- data.frame(dfpred, dflist[[i]][, 2])
}
colnames(dfpred) <- c('id', colnames(predicted.stacked)[-1])

# simple voting scheme
simple_vote <- function(df, min, max) {
    vote <- function(x) { # helper
        rev(order(tabulate(x)))[1]
    }
    apply(df[,c(min:max)], 1, vote)
}

all.voted <- data.frame(Id = dfpred$id, Response = simple_vote(dfpred, 2, 13))# LB ~ 0.66589
write.table(file = 'stacked/all-voted.csv', all.voted, quote = F, row.names = F, sep = ',')
xgb.voted <- data.frame(Id = dfpred$id, Response = simple_vote(dfpred,2,9)) # LB ~ 0.66587	
write.table(file = 'stacked/xgb-voted.csv', xgb.voted, quote = F, row.names = F, sep = ',') 
rf.voted <- data.frame(Id = dfpred$id, Response = simple_vote(dfpred,10,11))
write.table(file = 'stacked/rf-voted.csv', rf.voted, quote = F, row.names = F, sep = ',') 
xtr.voted <- data.frame(Id = dfpred$id, Response = simple_vote(dfpred,12,13))
write.table(file = 'stacked/xtr-voted.csv', xtr.voted, quote = F, row.names = F, sep = ',')
