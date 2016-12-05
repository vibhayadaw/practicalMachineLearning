##Data Cleaning and Preparation
The raw data comes in to files, training and testing.

```
train_in <- read.csv('./pml-training.csv', header=T)
validation <- read.csv('./pml-testing.csv', header=T)
```

###Data Partitioning

Since I’ll be predicting classes in the testing dataset, I’ll split the training data into training and testing partitions and use the pml-testing.csv as a validation sample. I’ll use cross validation within the training partition to improve the model fit and then do an out-of-sample test with the testing partition.

```
set.seed(127)
training_sample <- createDataPartition(y=train_in$classe, p=0.7, list=FALSE)
training <- train_in[training_sample, ]
testing <- train_in[-training_sample, ]
```
