library(dplyr); library(ggplot2)
# Ways to vary this simulation
# (1) change the signal
# (2) change N, the number of samples to train on
# (3) the noise (irreducible error)

signal = function(x) {
  # some mixture of sins and cosines (or not)
  # sin(x) + cos(3*x+pi/2) + sin(5*x+pi/4)  # in family
  # sin(x) + cos(3*x+pi/2) + sin(pi*pi*x)  # out of family (aperiodic)
   sin(x) + sin(x)*cos(x)/(0.1+x^2)  # out of family
  # x^2 + 3 - abs(x^3)  # out of family
}

N = 1000
noise = 2*rnorm(N)
period = 100
K = 10

dat = 
  data.frame(x = rnorm(N)) %>% as_tibble() %>%
  mutate(y = signal(x) + noise)

# # Plot the data alone
 ggplot(data=dat, aes(x=x,y=y)) + geom_point()

# Do a basis expansion of {sin(kx),cos(kx)} for k = 1 to 10
sincos = function(dat, period=2*pi, K=10) {
  data = dat
  for(i in 1:K) {
    data[[paste0("sin_",i)]] = sin(data$x*i*2*pi/period)
  }
  for(i in 1:K) {
    data[[paste0("cos_",i)]] = cos(data$x*i*2*pi/period)
  }
  data
}

# Learn a linear model
data = sincos(dat, period, K)
data = data %>% dplyr::select(-x)
lm = lm(data = data, y ~ .)
lm %>% summary()

# Learn a linear model, regularized
library(glmnet)  # Does regularization with (generalized) linear models
lasso = glmnet(x = data %>% dplyr::select(-y) %>% as.matrix(),
               y=data$y, family="gaussian") #Gaussian: normal dist around the linear model

# Look at attempted recovery of coefficients
ggplot(data=data.frame(x=c(0,-1:-K,1:K),values=lm$coefficients), aes(x=x, y=values)) + geom_col(width=0.2)

# Do prediction
datanew = data.frame(x=runif(1000)*10-5) %>% as_tibble()
datanew = sincos(datanew, period, K)
datanew[["yhat"]] = predict.lm(lm, datanew)
datanew[["ylasso"]] = predict(lasso, datanew %>% dplyr::select(-x,-yhat) %>% as.matrix(),
                              s = c(0.02)) #s = what the value for the lamda is; the level of sparsity 
datatruth = data.frame(x=datanew$x) %>% as_tibble() %>%
  mutate(y=signal(x))

# Plot
ggplot(data=dat, aes(x=x,y=y)) + geom_point() + 
  geom_line(data = datanew, aes(x=x,y=yhat,col="Prediction")) +
  geom_line(data = datanew, aes(x=x,y=ylasso,col="Lasso")) + 
  geom_line(data = datatruth, aes(x=x,y=y, col="Truth")) + ylim(c(-20,20))
  


### Your turn ###
# 1. uncomment sin(x) + cos(3*x+pi/2) + sin(5*x+pi/4) in signal(...) and comment
#    out the other return line.
#    Run the code and fit the function
# 2. Change dat to have x sampled from rnorm(...)*2. Then run twice with period: 
#    {5, 50}. What happens in each case? Why?

