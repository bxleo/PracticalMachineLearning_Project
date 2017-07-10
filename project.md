# Prediction Assignment
B. Li  
July 9, 2017  



## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. Different machine learning models will be built and cross-validated using the known data set ("pml-training.csv"), and predictions will be made on a new data set whose result is unknown ("pml-testing.csv"). 

## Load the Data

Download and load the data.


```r
nameKnownSet <- "pml-training.csv"
urlKnownSet <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
nameUnknownSet <- "pml-testing.csv"
urlUnknownSet <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (file.exists(nameKnownSet)) {
    knownSet <- read.csv(nameKnownSet, na.strings=c("NA","#DIV/0!",""))
} else { 
    download.file(urlKnownSet, nameKnownSet)
    knownSet <- read.csv(nameKnownSet, na.strings=c("NA","#DIV/0!",""))
}

if (file.exists(nameUnknownSet)) {
    unknownSet <- read.csv(nameUnknownSet, na.strings=c("NA","#DIV/0!",""))
} else { 
    download.file(urlUnknownSet, nameUnknownSet)
    unknownSet <- read.csv(nameUnknownSet, na.strings=c("NA","#DIV/0!",""))
}  
```


## Cleaning and Preparing the Data

Remove the first column, which is merely the row number. Also remove variables that are related with data acquisition (such as time stamps and window number).


```r
knownSet <- knownSet[, -c(1, 3:7)]
unknownSet <- unknownSet[, -c(1, 3:7)]
```

Remove the variables with too many NAs (>50%). 


```r
NACols1 <- sapply(knownSet, function(x) {if (mean(is.na(x)) > 0.5) {TRUE} else {FALSE}})
NACols2 <- sapply(unknownSet, function(x) {if (mean(is.na(x)) > 0.5) {TRUE} else {FALSE}})
knownSet <- knownSet[,!NACols1]
unknownSet <- unknownSet[,!NACols1]
```

Remove the last column of the unknown set, which is called "problem id".


```r
indexResult <- ncol(knownSet)
unknownSet <- unknownSet[, -indexResult]
```

## Training, Cross-Validation, Testing

First, split the known set into training set, validation set, and test set (60%, 20%, 20% for each). 


```r
library(caret)
set.seed(67439)
nrow_all <- nrow(knownSet)
split1 <- round(0.6 * nrow_all)
split2 <- round(0.8 * nrow_all)
temp <- sample((1:nrow_all))
trainSet <- knownSet[temp[1:split1], ]
validationSet <- knownSet[temp[(split1+1):split2], ]
testSet <- knownSet[temp[(split2+1):nrow_all], ]
```

Secondly, check if the variables are correlated with each other.


```r
cormat <- cor(knownSet[, -c(1,indexResult)])
nCorPairs <- (sum(abs(cormat) >= 0.7) - nrow(cormat))/2
nCorPairs
```

```
## [1] 38
```

There are 38 highly correlated pairs (correlation coefficient >= 0.7). Principal Component Analysis (PCA) is used to reduce dimension. Note that PCA is performed on the training set. When predicting the validation, test, and unknown sets, we will still use the principal components computed from the training set. 


```r
# the first column of trainSet is user_name
preprocessObj <- preProcess(trainSet[, -c(1,indexResult)], method = c("center", "scale", "pca"), thresh = 0.9)
preprocessObj
```

```
## Created from 11773 samples and 52 variables
## 
## Pre-processing:
##   - centered (52)
##   - ignored (0)
##   - principal component signal extraction (52)
##   - scaled (52)
## 
## PCA needed 18 components to capture 90 percent of the variance
```

This reduces the dimension of the numerical variables to 18. 


```r
ppTrainSet <- predict(preprocessObj, trainSet[, -c(1,indexResult)])
ppTrainSet$user_name <- trainSet$user_name
ppTrainSet$classe <- trainSet$classe
ppValidationSet <- predict(preprocessObj, validationSet[, -c(1,indexResult)])
ppValidationSet$user_name <- validationSet$user_name
ppValidationSet$classe <- validationSet$classe
ppTestSet <- predict(preprocessObj, testSet[, -c(1,indexResult)])
ppTestSet$user_name <- testSet$user_name
ppTestSet$classe <- testSet$classe
```

Using decision tree:


```r
fit_rpart <- train(classe ~., data = ppTrainSet, method = "rpart")
pred_validation_rpart <- predict(fit_rpart, newdata = ppValidationSet)
trainErr_rpart <- mean(predict(fit_rpart, ppTrainSet) == ppTrainSet$classe)
validationErr_rpart <- mean(pred_validation_rpart == ppValidationSet$classe)
```

Using random forest:


```r
library(randomForest)
fit_rf <- randomForest(classe ~., data = ppTrainSet)
pred_validation_rf <- predict(fit_rf, newdata = ppValidationSet)
trainErr_rf <- mean(fit_rf$predicted == ppTrainSet$classe)
validationErr_rf <- mean(pred_validation_rf == ppValidationSet$classe)
```

I also tried different methods such as SVM (see below for code). It runs very slow on my computer so the results are not shown here.


```r
fit_svmr <- train(classe ~., data = ppTrainSet, method = "svmRadial")
pred_validation_svmr <- predict(fit_svmr, newdata = ppValidationSet)
trainErr_svmr <- mean(fit_svmr$predicted == ppTrainSet$classe)
validationErr_svmr <- mean(pred_validation_svmr == ppValidationSet$classe)
```

The accuracies of training and validation are shown in the following table:


```r
accuracy <- data.frame(Train.Accu = c(trainErr_rpart, trainErr_rf), 
                       Validation.Accu = c(validationErr_rpart, validationErr_rf),
                       row.names = c("Decision Tree", "Random Forest"))
accuracy
```

```
##               Train.Accu Validation.Accu
## Decision Tree  0.3800221       0.3681529
## Random Forest  0.9659390       0.9633121
```

Therefore, **random forest is chosen**. The out-of-sample error is estimated using the test set:


```r
pred_test_rf <- predict(fit_rf, newdata = ppTestSet)
testErr_rf <- mean(pred_test_rf == ppTestSet$classe)
```

**The out-of-sample error is 0.9615189.**

## Predicting the Unknown Set

Preprocess the unknown using the principal components of the training set:


```r
ppUnknownSet <- predict(preprocessObj, unknownSet[, -1])
ppUnknownSet$user_name <- unknownSet$user_name
```

Predict the unknown set using random forest:


```r
pred_unknown_rf <- predict(fit_rf, newdata = ppUnknownSet)
pred_unknown_rf
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  A  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


