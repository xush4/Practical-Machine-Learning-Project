setwd("H:\\–ÌÍ…\\Rprogramming\\Github\\Practical-Machine-Learning-Project")
if (!file.exists("pml-training.csv")){
        setInternet2(use = TRUE)
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
}
if (!file.exists("pml-testing.csv")){
        setInternet2(use = TRUE)
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
}
trainData<-read.csv("H:\\–ÌÍ…\\Rprogramming\\Github\\Practical-Machine-Learning-Project/pml-training.csv",header = TRUE, sep = ",")
extestData<-read.csv("H:\\–ÌÍ…\\Rprogramming\\Github\\Practical-Machine-Learning-Project/pml-testing.csv",header = TRUE, sep = ",")
features <- c("roll_belt", "pitch_belt", "yaw_belt", "roll_arm", "pitch_arm", 
              "yaw_arm", "roll_forearm", "pitch_forearm", "yaw_forearm", "roll_dumbbell", 
              "pitch_dumbbell", "yaw_dumbbell")
class_feature <- c("classe")
trainData <- trainData[, c(features, class_feature)]
extestData <- extestData[, features]
library(caret)
library(rgl)

# BELT
plot3d(trainData$roll_belt, trainData$pitch_belt, trainData$yaw_belt, col = as.numeric(trainData$classe), 
       size = 1.5, xlab = "roll_belt", ylab = "pitch_belt", zlab = "yaw_belt")
plot3d(extestData$roll_belt, extestData$pitch_belt, extestData$yaw_belt, col = extestData$X, 
       size = 1, type = "s", add = TRUE)

# ARM
plot3d(trainData$roll_arm, trainData$pitch_arm, trainData$yaw_arm, col = as.numeric(trainData$classe), 
       size = 1.5, xlab = "roll_arm", ylab = "pitch_arm", zlab = "yaw_arm")
plot3d(extestData$roll_arm, extestData$pitch_arm, extestData$yaw_arm, col = extestData$X, 
       size = 1, type = "s", add = TRUE)

# FOREARM
plot3d(trainData$roll_forearm, trainData$pitch_forearm, trainData$yaw_forearm, 
       col = as.numeric(trainData$classe), size = 1.5, xlab = "roll_forearm", ylab = "pitch_forearm", 
       zlab = "yaw_forearm")
plot3d(extestData$roll_forearm, extestData$pitch_forearm, extestData$yaw_forearm, 
       col = extestData$X, size = 1, type = "s", add = TRUE)

# DUMBBELL
plot3d(trainData$roll_dumbbell, trainData$pitch_dumbbell, trainData$yaw_dumbbell, 
       col = as.numeric(trainData$classe), size = 1.5, xlab = "roll_dumbbell", 
       ylab = "pitch_dumbbell", zlab = "yaw_dumbbell")
plot3d(extestData$roll_dumbbell, extestData$pitch_dumbbell, extestData$yaw_dumbbell, 
       col = extestData$X, size = 1, type = "s", add = TRUE)
featurePlot(x = trainData[, features],
            y = trainData[, class_feature],
            plot = "density",
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"),
                          y = list(relation="free")),
            adjust = 1.5,
            pch = "|",
            auto.key = list(columns = 3))
# Data partition
set.seed(1234)
inTrainData <- createDataPartition(trainData$classe, p = 0.8, list = FALSE)
testData <- trainData[-inTrainData, ]
trainData <- trainData[inTrainData, ]
fitControl <- trainControl(method = "cv", number = 5)  # Low value in order to increase performance
library(plyr)

# Boosting with trees TRAIN
gbmModelFit <- train(classe ~ ., data = trainData, method = "gbm", trControl = fitControl, 
                     verbose = FALSE)
# Boosting with trees TEST
prediction <- predict(gbmModelFit, testData[, features])
confusionMatrix(prediction, testData$classe)
# Random forests TRAIN
rfModelFit <- train(classe ~ ., data = trainData, method = "rf", trControl = fitControl)
# Random forests TESTS
prediction <- predict(rfModelFit, testData[, features])
confusionMatrix(prediction, testData$classe)
predict(rfModelFit, extestData[, features])