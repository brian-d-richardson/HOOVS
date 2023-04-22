#---------- Set up ----------#

library(data.table)
library(devtools)
library(caret)
library(dplyr)
library(tictoc)
load_all()

#---------- Read in data and order outcome ----------#
data <- fread('C:/Users/brian/Documents/Grad School/BIOS 735/HOOVS/processed_data.csv')

y_factor_key <- data.frame(scaled = sort(unique(data$`B1:SATISFACTION W/LIFE`)),
                           y = factor(1:15, ordered = T))

data_ordered_y <- merge(y_factor_key, data, by.x = "scaled", by.y = "B1:SATISFACTION W/LIFE") %>% 
  arrange(`W1 Case Id`) %>%
  select(-c("scaled", "W1 Case Id"))

#---------- Create train and test / folds within CV ----------#

set.seed(5)
train_test_i <- createDataPartition(y = data_ordered_y$y, p = 0.6, list = F)

data_train <- data_ordered_y[train_test_i,]
data_test <- data_ordered_y[-train_test_i,]

k <- 5 #set number of folds 
fold_indicies <- createFolds(y = data_train$y, k=k)

#---------- Test run ----------#

lambda_grid <- seq(0, 0.2, 0.02)
J <- 15

tic("1 fold test")
res.ordreg <- ordreg.lasso(
  formula = y ~ .,
  data = data_train[-fold_indicies[[1]],],
  lambdas = lambda_grid
)
toc()

data.frame(lambda = res.ordreg$lambdas, 
           nonzero = res.ordreg$n.nonzero,
           kappa = res.ordreg$kappa)
