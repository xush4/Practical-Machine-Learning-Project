Practical Machine Learning Project Work - Weigh Lifting Dataset
========================================================

# Project Summary

The task of the project was to predict how well subjects performed weigh lifting excercises based on data collected form accelerometers attached to the person performing the exercises. The data set consists of data form six different people and the outcome is classified into five different categories. So, this is a supervised learning task, and the goal is to prduce a calssifier that orreclty classifies 20 samples provided as a testing set that needs to submitted for grading.


## Prediction algorithm selection

Random forrest algorithm usually performs rather well on a task like this, so that was chosen as the first algorithm to try. If the performance is not satisfacgtory, another algorithm will be tried

## Environment

We first intialize the environment by included the libraries used and by setting the seed for random number generator.


```r
require(dplyr)
```

```
## Loading required package: dplyr
```

```
## Warning: package 'dplyr' was built under R version 3.1.1
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
require(caret)
```

```
## Loading required package: caret
```

```
## Warning: package 'caret' was built under R version 3.1.1
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
require(randomForest)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(3434)
```

## Data processing

The data is included in two files provided.

First we read in the data, read.csv function does all the work with these two data sets:

- raw_data is the training data set
- test_data is the test set


```r
setwd("H:\\–ÌÍ…\\Rprogramming\\Github\\Practical-Machine-Learning-Project")
if (!file.exists("pml-training.csv")){
        setInternet2(use = TRUE)
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
}
if (!file.exists("pml-testing.csv")){
        setInternet2(use = TRUE)
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
}
raw_data <- read.csv("pml-training.csv")
test_data <- read.csv("pml-testing.csv")
```



We then remove features that have close to zero variance by using the near_zeros function of caret package. There variables have very little predictive value, so we should get rid of them

It's important to note that we have to take the same data mangling actions with both data sets.


```r
near_zeros <- nearZeroVar(raw_data[,-160])
raw_data_cleaned <- raw_data[, -near_zeros]
test_data_cleaned <- test_data[, -near_zeros]
```


There are also several variables that are missing data. Let's remove those since missing variables can cause problems with prediction algorithms, and we have plenty of data that can be used for prediction anyways. If this turns out to be a bad decision, we can always revisit it and impute some of the missing values.


```r
has_nas <- colSums(is.na(raw_data_cleaned)) > 0
raw_data_cleaned <- raw_data_cleaned[, !has_nas]
test_data_cleaned <- test_data_cleaned[, !has_nas]
```


Some of the variables seemed to be irrelevant for prediction, so we should remove them as well.

These varibales are: X, user_name, timestamps, num_window (six first variables)


```r
raw_data_cleaned <- raw_data_cleaned[, 7:59]
test_data_cleaned <- test_data_cleaned[, 7:59]
```


After all the cleaning work, we are left with 52 variabels and will use those for predicting the outcome.

## Creating data sets for cross-validation

We split training set to test and validation parts (70% - 30% split), so we can get a good out of bag error estimate - estimate.


```r
in_train <- createDataPartition(raw_data_cleaned$classe, p = 0.7, list=FALSE)
training <- raw_data_cleaned[in_train, ]
testing <- raw_data_cleaned[-in_train, ]
```

## Prediction

Well use care train function from caret package to run cross valdiated prediction. We use 4 as the number for cross calidation partitions instead of standard ten. The data set is rather large and this will run faster on a laptop we're using.

After training the model, we'll predict the outcome values for the validation set (30% of the orignal training dataset that we set aside to estiamte the performance of the model.)


```r
fit <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method = "cv", number = 4))
test_pred <- predict(fit, newdata=testing)
```

We then use the a function from caret pacakge to create a confusion matrox of the results so we can inspect the performance of the mode.


```r
confusionMatrix(test_pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    3    0    0    0
##          B    2 1133    5    0    1
##          C    0    2 1019   10    4
##          D    0    1    2  952    3
##          E    0    0    0    2 1074
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.992         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.995    0.993    0.988    0.993
## Specificity             0.999    0.998    0.997    0.999    1.000
## Pos Pred Value          0.998    0.993    0.985    0.994    0.998
## Neg Pred Value          1.000    0.999    0.999    0.998    0.998
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.173    0.162    0.182
## Detection Prevalence    0.285    0.194    0.176    0.163    0.183
## Balanced Accuracy       0.999    0.997    0.995    0.993    0.996
```

We see the accuracy of the model (out ot sample) is 99.39% so we are expecting an error of ~ 0.6%. Pretty good for a model, so we don't have to go back and tweak the parameters of cross validation or the model itself.

Next, let's generate the project test set predictions with this classifier.


```r
real_preds <- predict(fit, newdata=test_data_cleaned)
```

And finally write them to a file with a function provided in the project.


```r
# function for writing the output files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

# write the predictons generated in the previsou step into files.
pml_write_files(real_preds)
```

After feeding the created files to the project grading system, we got all predictions right (20/20) so we are very satisfied with the result.

# Results

The random forrest model we buil to predict the outcome of activities performed perfectly with the 20 sample test set provided in the project.

We used 52 vriables to buil the mode with 4 fold cross validation. The out of sample error estimate was 0.6%.
