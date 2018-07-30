library(dplyr); library(data.table); library(purrr);
library(tidyr); library(stringr); library(ggplot2)


## Load
dat =
  fread("/Users/michellehsu/Desktop/CMU/Spring 2018/95-845 Applied Analytics-ML Pipeline/Reference/AA_MLP_intro_poll_redacted.csv") %>%
  as_tibble()
dat %>% View

## Show
dat %>% summary()
ggplot(data=dat) + geom_histogram(aes(x=Theory),binwidth=0.5) + theme_bw()
ggplot(data=dat) + geom_histogram(aes(x=Design),binwidth=0.5) + theme_bw()
ggplot(data=dat) + geom_histogram(aes(x=EndtoEnd),binwidth=0.5) + theme_bw()

# LOOCV with a decision tree: predicting use of git
# install.packages("rpart"); install.packages("rpart.plot")
library(rpart); library(rpart.plot)
folds =  # tidyr preparation
  crossing(data.frame(iter=1:dim(dat)[1]) %>%
             as_tibble(),
           dat %>% mutate(id = 1:n())) %>%
  mutate(train= c("train","test")[(iter==id)+1]) %>%
  nest(-train,-iter) %>%
  arrange(iter,desc(train)) %>%
  spread(train, data)
folds

results =  # purrr map on each row using map
  folds %>%
  mutate(tree = map(train, ~ rpart(git ~ ., 
                                   data = .x %>% mutate(git=as.factor(git)) %>% dplyr::select(-Time, -id),
                                   control=rpart.control(minsplit = 4)))) %>%
  mutate(prediction = map2(tree, test, ~ predict(.x, newdata=.y, type="prob"))) %>%
  mutate(prediction = map_dbl(prediction, ~ .x[[1]])) %>%
  mutate(truth = map_dbl(test, ~ .x$git))
results
  
# Inspect one tree:
tree1 = results$tree[[1]]; tree1
tree2 = results$tree[[2]]; tree2
rpart.plot(tree1)
rpart.plot(tree1, branch.type=5)


### Your turn ###
# The code above did LOOCV (leave one out cross validation)
# Write code that does 5-fold cross-validation.
# Note that k-fold cross-validation is a type of "map reduce"
# - map: data --> k folds of (train/test) data
# - apply: make a decision tree, apply it to test set for predictions
# - reduce: collect performance statistics across folds