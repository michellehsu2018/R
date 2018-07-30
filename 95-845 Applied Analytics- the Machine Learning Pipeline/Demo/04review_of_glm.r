#install.packages("readr")
library(readr)

### Mushroom classification example ###
#read_table(file="http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data")
mush = read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv")
mush %>% count()
mush = mush[sample(1:nrow(mush)),]
mushtrain = mush[1:5000,]
mushtest = mush[-(1:5000),]
mushtrain %>% select(type) %>% table()
mushtrain %>% select(type,cap_shape) %>% table()
mushtrain %>% select(type,cap_shape,cap_surface) %>% table()

?glm
#glm(formula, family = gaussian, data, weights, subset, 
    #na.action, start = NULL, etastart, mustart, offset,
    #control = list(...), model = TRUE, method = "glm.fit",
    #x = FALSE, y = TRUE, contrasts = NULL, ...)
##formula:a symbolic description of the model to be fitted. The details of model specification are given under ‘Details’.
##family:a description of the error distribution and link function to be used in the model.

#with(data, expr, ...)
##data:data to use for constructing an environment. For the default with method this may be an environment, a list, a data frame, or an integer as in sys.call.
##expr:expression to evaluate
lr = with(mushtrain, glm(type=="e"~cap_shape, family = binomial("logit")))
lr %>% summary()
lr %>% str()
lr %>% summary() %>% coef()
?predict.glm
#predict.glm {stats}
##Obtains predictions and optionally estimates standard errors of those predictions from a fitted generalized linear model object.
#predict(object, newdata = NULL,
       #type = c("link", "response", "terms"),
       #se.fit = FALSE, dispersion = NULL, terms = NULL,
       #na.action = na.pass, ...)

##object: a fitted object of class inheriting from "glm".
##newdata: optionally, a data frame in which to look for variables with which to predict. 
         ##If omitted, the fitted linear predictors are used.

predictions = data.frame(preds=(lr %>% predict(mushtest, type="response")))
mushtest %>%
  select(type) %>%
  bind_cols(predictions) %>%
  table() %>%
  data.frame() %>% 
  tbl_df() %>%
  group_by(preds) %>%
  mutate(testfreq = sum(Freq)) %>%
  mutate(testfreq = Freq/testfreq) %>%
  ungroup() %>% filter(type=='e') %>%
  select(preds,testfreq)

# comment out parts of this chain with ctrl-shift-c to see what it is computing

# bonus: lasso-regularized logistic regression
install.packages("glmnet")
library(glmnet)
library(tidyr)
mushmm = model.matrix(type~., mush[,-17]) # turns categorical vars into indicators
mushmmtrain = mushmm[1:5000,]
mushmmtest = mushmm[-(1:5000),]
lr1 = glmnet(x = mushmmtrain, y=mushtrain$type, family="binomial")
str(lr1)
which(coef(lr1)[,20] > 0)
which(coef(lr1)[,10] > 0)

data.frame(preds = lr1 %>% predict(mushmmtest, s=0.2, type="response"),
           y = mushtest$type) %>% tbl_df() %>%
  table() %>% data.frame() %>% tbl_df() %>%
  transmute(type=y, preds=X1, Freq=Freq)%>%
  arrange(preds) %>%
  group_by(preds) %>%
  mutate(testfreq = sum(Freq)) %>%
  mutate(testfreq = Freq/testfreq) %>%
  ungroup() %>%
  filter(type=='p') %>%
  select(preds,testfreq)



### Your turn ###
# add a column "x_and_s" to mush that is 1 if
#    cap_shape equals "x" and odor equals "s"
# filter the mush data to keep only those where "x_and_s" is not 1
# restore mush to the original data set if necessary


# use logistic regression and the training data (mushtrain) to predict
#   whether a mushroom is edible, based on: cap_surface, cap_color, and odor.
#   Compare your predictions to the real outcome on the first 100 examples in mushtest

# report which features from this model are significant

# use R to compute the percentage of edible mushrooms and poisonous ones
