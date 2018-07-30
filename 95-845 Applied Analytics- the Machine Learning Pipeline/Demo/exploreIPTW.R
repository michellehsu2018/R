# Inverse probability of treatment weighting example

# Goal: we want to estimate causal effect of treatment U on outcome Y
# Method: inverse weighting, doubly robust estimation
# Why: we are worried that confounders V will result in corrupted estimates
#      if we do not take them into account
# How: we have two principal way to account for them:
#   1. inverse weighting/propensity scoring
#   2. covariate adjustment/stratification
# Assumptions: unconfoundedness, consistency, common support

# install.packages("twang")
library(twang)
library(dplyr)


# Y, U, V
N = 1000

getData = function(N) {
  # V confounders
  V1 = rnorm(N, mean = 5)
  V2 = rpois(n = N, lambda = 20)
  
  # U in {0, 1}, but dependent on V
  Utemp = exp(8 - V1 - V2/10)
  Uprob = Utemp / (1 + Utemp)
  U = Uprob > runif(N)
  
  # Y = 4*U + V + rnorm()
  #Y = 4*U + V1^2 + V2/10 + rnorm(N, sd=1)
  Y = 4*U + V1 + V2/10 + rnorm(N, sd=1)
  
  # Y, U, V1, V2 in a data frame
  data = data.frame(Y=Y, U=U, V1=V1, V2=V2) %>% as_tibble() 
  data
}
v= c()
w = c()
for(i in 1:5) {
  data = getData(N)
  
  # now we want to recover the causal effect of U on Y: 4
  
  # with weights
    ip = ps(U ~ V1 + V2, data = as.data.frame(data),
           n.trees = 1000, stop.method = "es.mean", verbose=F)
  # # (note twang doesn't like tibbles)
  weights = get.weights(ip, stop.method = "es.mean")  # already inverted
  
  # without weights
  weights.no = rep(1,N)
  
  # Let's learn a linear model for Y (doubly robust estimation)
  lr = lm(Y ~ ., data=data, weights = weights)
  summary(lr)
  print(lr$coefficients[2])
  v = c(v, lr$coefficients[2])
  
  lr.no = lm(Y ~ ., data=data, weights = weights.no)
  summary(lr.no)
  print(lr.no$coefficients[2])
  w = c(w, lr.no$coefficients[2])
}
hist(v)
hist(w)

### Your turn: 
# Show (through double-robustness) that given a ground truth linear model,
# estimation of beta_u no longer requires the inverse weighting to get
# an unbiased estimate.
# To do so:
# - Modify the equation for Y by changing the V1^2 term to a V1 term.
# - Rerun the function definition getData to update the data generator
# - Rerun the estimation loop