# Predicting breast cancer using Xgboost
Woosang Lee  
Feb 1st, 2016  

In this document, I will use a data about breast cancer diagnosis. This data is extracted and processed from [original data from Wisconsin University](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). It is provided from Packt Publishing to practice a example in the book **Machine Learning with R - Second Edition**. If you want to get data used in this document, visit [official Packt website](https://www.packtpub.com/big-data-and-business-intelligence/machine-learning-r-second-edition).





*This document is written at feb 01, 2016 and Not yet edited.*

##Library and Data Preparing
In this document, caret packaged is used to predict breast cancer.


```r
# Load Library
library(xgboost)
require(Matrix)
require(data.table)
library(caret)

# Data Load
rawData<- read.csv("Data/wisc_bc_data.csv", header = TRUE, stringsAsFactors = TRUE)
dataDim<-dim(rawData)
rawData <- data.table(rawData, keep.rownames = F)
rawData[,id:=NULL]
```

This data has **32 features** and **569 observations**. For performance, this data is converted to datatable.

Before make model, we need to seperate raw data into training and testing data. **createDataPartition** is useful for this process. Below code show allocating 80% of data to training set and 20% to testing set. After devide dataset, we need to make data matrix and label.


```r
# Seperate data into training set and test set
set.seed(12323)
inTrain<-createDataPartition(rawData$diagnosis, p=0.8, list=FALSE)
trainSet<-rawData[inTrain]
testSet<-rawData[-inTrain]

# Make data matrix and label for train and test set
trainLabel <- trainSet$diagnosis == "B"
trainFeature <- trainSet[,diagnosis:=NULL]
train_matrix <- data.matrix(trainFeature)

testLabel <- testSet$diagnosis == "B"
testFeature <- testSet[,diagnosis:=NULL]
test_matrix <- data.matrix(testFeature)

# Feature and label for xgboost
dtrain <- xgb.DMatrix(data = train_matrix, label = trainLabel)
```

There are 456 observations in train set, and 113 in test set. **dtrain** variable is for train, but can be skipped.

##Build Model
Xgboost will be used to build model. To make model, you can use `train` function.


```r
# xgboost train
bst <- xgboost(data = dtrain, max.depth = 5,
               eta = 2, nthread = 2, nround = 50, objective = "binary:logistic")
```

After build model, we can predict by using `predict` function. `confusionMatrix` function provide best information about result of model. We will see that information of better model at the last of this document.


```r
# Predict result
expecting<- predict(bst, test_matrix)
result <- confusionMatrix(expecting>0.5, testLabel)
```

Accuracy of this model is **99.12%**.

