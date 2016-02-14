##########################################################################
# Load and preprocess the data
##########################################################################
library(caret)
library(mlr)
# load data
setwd('~/leave_academia/kaggle/prudential')
train.data <- read.csv(file = 'train.csv', header = T, stringsAsFactors = F)
test.data <- read.csv(file = 'test.csv', header = T, stringsAsFactors = F)

# compute NAs
train.na <- apply(train.data, 2, function(x) sum(is.na(x)))/dim(train.data)[1]
test.na <- apply(test.data, 2, function (x) sum(is.na(x)))/dim(test.data)[1]

# target frequencies
target.data <- factor(train.data$Response, ordered=T)
target.rfq <- summary(target.data)/length(target.data) # relative frequencies

train.data <- data.frame(id = train.data$Id, response = train.data$Response, train.data[,-c(1,128)])
test.data <- data.frame(id = test.data$Id, response = NA, test.data[,-1])

# merge train and test
all.data <- rbind(train.data, test.data)
rm(train.data, test.data)

##########################################################################
### List variables by type
##########################################################################
var.cont.nms <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
                  "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                  "Insurance_History_5", "Family_Hist_2", "Family_Hist_3",
                  "Family_Hist_4", "Family_Hist_5")

var.factor.nms <- c("Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5",
"Product_Info_6", "Product_Info_7", "Employment_Info_2", "Employment_Info_3",
"Employment_Info_5", "InsuredInfo_1", "InsuredInfo_2", "InsuredInfo_3",
"InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",
"Insurance_History_1", "Insurance_History_2", "Insurance_History_3",
"Insurance_History_4", "Insurance_History_7", "Insurance_History_8",
"Insurance_History_9", "Family_Hist_1", "Medical_History_2", "Medical_History_3",
"Medical_History_4", "Medical_History_5", "Medical_History_6", "Medical_History_7",
"Medical_History_8", "Medical_History_9", "Medical_History_11", "Medical_History_12",
"Medical_History_13", "Medical_History_14", "Medical_History_16", "Medical_History_17",
"Medical_History_18", "Medical_History_19", "Medical_History_20", "Medical_History_21",
"Medical_History_22", "Medical_History_23", "Medical_History_25", "Medical_History_26",
"Medical_History_27", "Medical_History_28", "Medical_History_29", "Medical_History_30",
"Medical_History_31", "Medical_History_33", "Medical_History_34", "Medical_History_35",
"Medical_History_36", "Medical_History_37", "Medical_History_38", "Medical_History_39",
"Medical_History_40", "Medical_History_41")

##########################################################################
### Create Features from Product_Info_2
##########################################################################
# Product_Info_2 to numeric and split the letter
all.data$Product_Info_2_Letter <- sapply(all.data$Product_Info_2, substr, 1, 1)
all.data$Product_Info_2_Letter <- factor(all.data$Product_Info_2_Letter)
levels(all.data$Product_Info_2_Letter) <- c(1:5)
all.data$Product_Info_2_Letter <- as.numeric(all.data$Product_Info_2_Letter)

all.data$Product_Info_2_Number <- sapply(all.data$Product_Info_2, substr, 2, 2)
all.data$Product_Info_2_Number <- as.numeric(all.data$Product_Info_2_Number)

all.data$Product_Info_2 <- factor(all.data$Product_Info_2)
levels(all.data$Product_Info_2) <- c(1:length(levels(all.data$Product_Info_2)))
all.data$Product_Info_2 <- as.numeric(all.data$Product_Info_2)

# update factor list
var.factor.nms <- c(var.factor.nms, 'Product_Info_2_Letter', 'Product_Info_2_Number')

##########################################################################
### Impute continuous with NAs: improved LB score
##########################################################################

# First examine continuous variables with NAs
continuous.nms <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI", "Employment_Info_1", "Employment_Info_4",
                    "Employment_Info_6", "Insurance_History_5", "Family_Hist_2",
                    "Family_Hist_3", "Family_Hist_4", "Family_Hist_5")
all.na <- apply(all.data, 2, function(x) sum(is.na(x)))/dim(all.data)[1]
varna.nms <- names(all.na[which(all.na>0)])

###----------------------------------
### Rules for Imputation
### Remove Medical History 10, 32 and Employment Info 4 (few observations)
### Employment Info 1 by median
### 
### Insurance_History_5 ctree
### Employment_Info_6 ctree
### Medical_History_15 ctree
### Medical_History_24 ctree
### Fam_Hist_5 ctree
### Fam_Hist_3 ctree
### Medical_History_1 ctree
### Fam_Hist_4,2 ctree
###----------------------------------

toremove <- c('Medical_History_10', 'Medical_History_32', 'Employment_Info_4')
all.data$Medical_History_10 <- NULL
all.data$Medical_History_32 <- NULL
all.data$Employment_Info_4 <- NULL

# update continuous variables
var.cont.nms <- c("Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
                  "Employment_Info_1", "Employment_Info_6",
                  "Insurance_History_5", "Family_Hist_2", "Family_Hist_3",
                  "Family_Hist_4", "Family_Hist_5")
varna.nms <- varna.nms[-c(3,11,14)]
# impute by median
all.data$Employment_Info_1[is.na(all.data$Employment_Info_1)] <- median(all.data$Employment_Info_1,
                                                                        na.rm = T)
# preprocess by x <- log(1+x)
logtrans <- c('Medical_History_15', 'Medical_History_24')
all.data[logtrans] <- log(1 + all.data[logtrans])
all.data['Insurance_History_5'] <- log(1e-4 + all.data['Insurance_History_5'])
inshist5.scaler <- preProcess(all.data['Insurance_History_5'], method = c('range'))
all.data['Insurance_History_5'] <- predict(inshist5.scaler, all.data['Insurance_History_5'])
# YeoJohnson + range
medhist1.scaler <- preProcess(all.data['Medical_History_1'], method = c('YeoJohnson', 'range'))
all.data['Medical_History_1'] <- predict(medhist1.scaler, all.data['Medical_History_1'])

# impute by ctree (better results on rpart)
toimpute <- c('Insurance_History_5', 'Employment_Info_6', 'Medical_History_15',
              'Medical_History_24', 'Family_Hist_5', 'Family_Hist_3',
              'Medical_History_1', 'Family_Hist_4', 'Family_Hist_2')
for(v in toimpute) {
    print(paste('Imputing', v))
    indx <- is.na(all.data[v])
    lrn.impute <- makeLearner('regr.ctree')
    impute.task <- makeRegrTask(data = all.data[!indx,-c(1,2)], target = v)
    trained <- train(lrn.impute, impute.task)
    vindx <- which(colnames(all.data) == v)
    predicted <- predict(trained, newdata = all.data[indx, -c(1,2,vindx)])
    save(predicted, file = paste(paste('imputation/imputed-', v, sep=''),'.RData',sep=''))
    all.data[indx, v] <- predicted$data['response']
}

save(all.data, file = 'imputation_alldata.RData')
##########################################################################
# Helper function for exploratory analysis and check effects of imputation
generate_cont_plots <- function() {
    for(i in var.cont.nms) { # loop over qplots; note use of print()
    x11()
    print(qplot(all.data[,i], data = all.data, geom = 'density', main = i))
    }
}

##########################################################################
### New feature
##########################################################################
# sum medical keywords
all.data$Total_Medical_Keywords <- apply(all.data[, grep('Medical_K', colnames(all.data))],
                                         1, sum)
# add to continuous
var.cont.nms <- c(var.cont.nms, 'Total_Medical_Keywords')

##########################################################################
### Scale features not imputed
##########################################################################

# Continuous features ----------------------------------
# scaler features according to method
var.minmax.nms <- c('Total_Medical_Keywords')
var.stdscal.nms <- c('Ht', 'BMI', 'Wt')

rg.scaler <- preProcess(all.data[var.minmax.nms], method = c('range'))
all.data[var.minmax.nms] <- predict(rg.scaler, all.data[var.minmax.nms])

std.scaler <- preProcess(all.data[var.stdscal.nms], method = c('center', 'scale'))
all.data[var.stdscal.nms] <- predict(std.scaler, all.data[var.stdscal.nms])

# discrete variables

save(all.data, target.data, file = 'preprocessed.RData')
