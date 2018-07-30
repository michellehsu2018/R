# Data come from SPECT scan of heart to look at normal or
# abnormal contraction of the left ventricle of the heart.
# For more details, see:
# http://pages.cs.wisc.edu/~dyer/cs540/handouts/kurgan-cardiac2001.pdf

library(dplyr)

spectTrainUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train"
spectTestUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test"

spectTrain = read.csv(file = spectTrainUrl, header=F) %>% tbl_df()
spectTest = read.csv(file = spectTestUrl, header=F) %>% tbl_df()


### SVMs
library(e1071)
svmModel = svm(V1 ~ . , spectTrain, kernel="polynomial", degree= 6, coef=0)
svmPr = data.frame(svmpr = predict(svmModel, spectTest),
                   truth = spectTest$V1==1) %>% tbl_df()
svmPr %>% mutate(svmpr = svmpr>0.5) %>% table()

library(ROCR)
prediction(predictions = svmPr$svmpr, labels = spectTest$V1==1) %>%
  performance(measure = "tpr",x.measure = "fpr") %>% plot(col=4)


### Your turn: SVMs

# Change the kernel function to polynomial. See the ?svm file to see how to do this.
# (Note you almost certainly want to set coef0 to be non-zero to enable learning of lower order
#   polynomials.)
#   e.g. kernel="polynomial", degree=6, coef=1 --> 0 to 6th order 
#   e.g. kernel="polynomial", degree=6, coef=0 --> 6th order exactly
# How does your performance change?
# Plot the predictions for radial and polynomial kernels in a scatterplot to assess agreement.





### Survival analysis: KM
library(survival)
?lung
ldat = lung %>% tbl_df()
ldat
km = Surv(ldat$time, ldat$status==2) %>% #indicate the event did occur
  (function(x) survfit(x ~ 1, data=ldat))(.) #use object to regress on nothing
# Kaplan-Meier curve
plot(km, xlab= "Days",ylab="P(survive)", mark.time=T)
km



### Cox model
cm = Surv(ldat$time, ldat$status==2) %>%
  (function(x) coxph(x ~ age + sex +inst + #x: survival object
                       meal.cal + wt.loss, data=ldat))(.)
cm
#sex 0.634859  #being a female, you are going to die at a lower rate : hazard *  0.634859 

### Regularized Cox model
library(glmnet)
lm = ldat %>% na.omit() %>%
  (function(x) glmnet(x %>% select(-time,-status) %>%
                        as.data.frame() %>% as.matrix(),
                      Surv(x$time, x$status==2),
                      family="cox"))(.)
coef(lm, s=0.1)

#How to choose s? Consider sparsity you want with cross-validation
cvlm = ldat %>% na.omit() %>%
  (function(x) cv.glmnet(x %>% select(-time,-status) %>%
                           as.data.frame() %>% as.matrix(),
                         Surv(x$time, x$status==2),
                         family="cox"))(.)
plot(cvlm)


### Your turn. Plot the Kaplan Meier curves one for each sex.
# Does there appear to be a difference in survival?
# Run a Cox model with only sex. Does this help you answer the previous question?
# What concerns might you have about the relationship identified between sex and survival?

#Female
km_female = Surv(ldat$time, ldat$status==2 & ldat$sex == 2) %>% #indicate the event did occur and is female
  (function(x) survfit(x ~ 1, data=ldat))(.) 
# Kaplan-Meier curve for female
plot(km_female, xlab= "Days",ylab="P(survive)", mark.time=T)

#male
km_male = Surv(ldat$time, ldat$status==2 & ldat$sex == 1) %>% #indicate the event did occur and is female
  (function(x) survfit(x ~ 1, data=ldat))(.) 
# Kaplan-Meier curve for female
plot(km_male, xlab= "Days",ylab="P(survive)", mark.time=T)

### Cox model only one sex
cm_sex = Surv(ldat$time, ldat$status==2) %>%
  (function(x) coxph(x ~ sex , data=ldat))(.)
cm_sex

#The number of female and male object is not balanced
#Censorships
#Ignoring other confoundings
sum(ldat$sex==1) #male: 138
sum(ldat$sex==2) #female: 90
