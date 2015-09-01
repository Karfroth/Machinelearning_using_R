# Predicting breast cancer using Random Forest
Woosang Lee  
Sep 1st, 2015  

In this document, I will use a data about breast cancer diagnosis. This data is extracted and processed from [original data from Wisconsin University](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). It is provided from Packt Publishing to practice a example in the book **Machine Learning with R - Second Edition**. If you want to get data used in this document, visit [official Packt website](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-r-second-edition).





*This document is written at Sep 01, 2015 and Not yet edited.*

##Library and Data Preparing
In this document, caret packaged is used to predict breast cancer.


```r
# Load Library
library(caret)
library(randomForest)

# Data Load
rawData<- read.csv("Data/wisc_bc_data.csv", header = TRUE, stringsAsFactors = TRUE)
dataDim<-dim(rawData)
```

This data has **32 features** and **569 observations**. 

Before make model, we need to seperate raw data into training and testing data. **createDataPartition** is useful for this process. Below code show allocating 80% of data to training set and 20% to testing set.


```r
# Seperate data into training set and test set
set.seed(12323)
inTrain<-createDataPartition(rawData$diagnosis, p=0.8, list=FALSE)
trainSet<-rawData[inTrain,-1]
testSet<-rawData[-inTrain,-1]
```

There are 456 observations in train set, and 113 in test set.

##Build Model
Random Forest will be used to build model. In this Document, we will use two method. One is Bootstrap, and another is Repeated Cross Validation. 10 is asigned to `number` and `repeats`. If you need more information, check trainControl document. To make model, you can use `train` function.


```r
# random forest with Bootstrap
ctrl.boot<-trainControl(method="boot", number=10, repeats = 10)
grid_rf<-expand.grid(.mtry=c(2,4,6,8,16))
set.seed(12323)
modFit.boot<- train(x=trainSet[,-1], y=trainSet[,1], method = "rf",
                        tuneGrid = grid_rf, ntree=200)

# random forest with Repeated Cross Validation
ctrl.rcv<-trainControl(method="repeatedcv", number=10, repeats = 10)
set.seed(12323)
modFit.rcv<- train(x=trainSet[,-1], y=trainSet[,1], method = "rf", trControl=ctrl.rcv,
               tuneGrid = grid_rf, ntree=200)
```

After build model, we can predict by using `predict` function. `confusionMatrix` function provide best information about result of model. We will see that information of better model at the last of this document.


```r
# Predict result
expecting.rcv<- predict(modFit.rcv, testSet)
expecting.boot<- predict(modFit.boot, testSet)
result.rcv <- confusionMatrix(expecting.rcv, testSet$diagnosis)
result.boot <- confusionMatrix(expecting.boot, testSet$diagnosis)
betterModelName<-ifelse(result.rcv$overall[1]>result.boot$overall[1], 
                        "Repeated Cross Validation", "Bootstrap")
```

Accuracy of Random Forest with Repeated Cross Validation is **97.35%** and Accuracy of Random Forest with Bootstrap is **96.46%**.

Therefore Repeated Cross Validation is better model in this time. Below is overview of Repeated Cross Validation.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  B  M
##          B 71  3
##          M  0 39
##                                           
##                Accuracy : 0.9735          
##                  95% CI : (0.9244, 0.9945)
##     No Information Rate : 0.6283          
##     P-Value [Acc > NIR] : <2e-16          
##                                           
##                   Kappa : 0.9423          
##  Mcnemar's Test P-Value : 0.2482          
##                                           
##             Sensitivity : 1.0000          
##             Specificity : 0.9286          
##          Pos Pred Value : 0.9595          
##          Neg Pred Value : 1.0000          
##              Prevalence : 0.6283          
##          Detection Rate : 0.6283          
##    Detection Prevalence : 0.6549          
##       Balanced Accuracy : 0.9643          
##                                           
##        'Positive' Class : B               
## 
```
