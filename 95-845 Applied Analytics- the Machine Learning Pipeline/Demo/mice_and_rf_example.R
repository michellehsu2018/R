setwd("C:/Users/Jeremy/Box Sync/18 Spring/aa/")
#mice: Multivariate Imputation By Chained Equations ==> doing EM algorithm multiple times to impute missing data
#mice returns different data set with different imputed missing data filled(set by parameter)
#The package creates multiple imputations (replacement values) for multivariate missing data. 
##The method is based on Fully Conditional Specification, where each incomplete variable is imputed by a separate model. 
##The MICE algorithm can impute mixes of continuous, binary, unordered categorical and ordered categorical data. 
##In addition, MICE can impute continuous two-level data, and maintain consistency between imputations by means of passive imputation.
##Many diagnostic plots are implemented to inspect the quality of the imputations.##
#rf: random forest
### Load helper files ###
loadlibs = function(libs) {
  for(lib in libs) {
    class(lib)
    if(!do.call(require,as.list(lib))) {install.packages(lib)}
    do.call(require,as.list(lib))
  }
}
libs = c("tidyr","magrittr","purrr","dplyr","stringr","readr","data.table")
loadlibs(libs)

### Example for part 2

mush = read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv")
mush %>% count()
mush = mush[sample(1:nrow(mush)),]  # permute rows: rearrange the rows
mush = mush %>% select(-veil_type)  # no information, just causes problems
mushtrain = mush[1:5000,]
mushtest = mush[-(1:5000),]
typeof(mushtrain)
mushtrain %>% select(type) %>% table()
mushtrain %>% select(type,cap_shape) %>% table()
mushtrain %>% select(type,cap_shape,cap_surface) %>% table()

# Corrupt the data by generating NAs randomly (MCAR)
for (i in 1:10000) {
  mush[sample(dim(mush)[1],1), sample(dim(mush)[2],1)] = NA
}
namush = mush; # rename
rm(mush) #remove mush 
# dim(mush)[1]==>the rows in the mush 
# Run multiple imputation (though just single for our exposition). **Note, do not use label in imputation**
library(mice)  # you may try others if desired, e.g. 'amelia'  #you should not include the outcome label
# Notice you need to convert all character columns to factor columns
mdat = mice(namush %>% select(-type) %>% mutate_if(is.character, as.factor),m=1,maxit = 1)  
# Also, for robust analyses you set m>10 (multiple imputation and maxit > 5 (burnin), but we will just use m=1 (single imputation) and maxit=1
# m: how many imputed data set you want mice to give; maxit: how many back and forth you want it to reimpute missing data; giving the number of iterations

idat = complete(mdat) %>% as_tibble()
#complete:Turns implicit missing values into explicit missing values.
# idat = as.data.frame(lapply(idat, function (x) if (is.factor(x)) factor(x) else x)) %>% tbl_df()

names(idat) = lapply(names(idat), paste0, "_imputed")
#lapply: Apply a Function over a List or Vector
#Rename those columns in case you want to column append

##Find out which columns has the missing value! 
missing = apply(namush,FUN = (function(x) any(is.na(x))),MARGIN = 2)
#any:Given a set of logical vectors, is at least one of the values true?
#MARGIN is a variable defining how the function is applied: when MARGIN=1, it applies over rows, whereas with MARGIN=2, it works over columns. 
#Note that when you use the construct MARGIN=c(1,2), it applies to both rows and columns

#Exclude the type column since it is the outcome label
missing = missing[-which(names(missing)=="type")]
#which:Give the TRUE indices of a logical object, allowing for array indices.

idat = idat[,paste0(names(missing %>% t() %>% data.frame()),"_imputed")]
imputedmush = data.frame(type = namush$type) %>% bind_cols(idat) %>% as_tibble()

library(rpart); library(rpart.plot)
tree = rpart(imputedmush, formula = type ~ .)
rpart.plot(tree, branch.type=5)


# Example random forest
#
library(randomForest)
forest = randomForest(formula = type ~ ., data=imputedmush %>% filter(!is.na(type)))
forest$importance
predict(forest, imputedmush, type = "prob") %>% head()


# Example adaboost
library(ada)
aforest = ada(formula = type ~ .,
              data=imputedmush %>% filter(!is.na(type)),
              iter=10)
predict(aforest, imputedmush,
        type = "probs") %>% head()


# Example gradient boosted forest
library(gbm)
gforest = gbm(formula = type=="e" ~ ., data=imputedmush %>% filter(!is.na(type)))
predict(gforest, imputedmush,
        n.trees=gforest$n.trees,
        type = "response") %>% head()

