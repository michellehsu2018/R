#install.packages("devtools")
#devtools::install_github("rstudio/keras")
library(keras)#all data are in matrix form
library(dplyr); library(readr)

N = 1000

# Data generator part 1
getData = function(N) {
  # V confounders
  V1 = rnorm(N, mean = 5)
  V2 = rpois(n = N, lambda = 20)
  
  # U in {0, 1}, but dependent on V
  Utemp = exp(8 - V1 - V2/10)
  Uprob = Utemp / (1 + Utemp)
  U = Uprob > runif(N)
  
  # Y = 4*U + V + rnorm()
  Y = 4*U + V1^2 + V2/10 + 0.1*V1*V2^2 + rnorm(N, sd=10)
  
  # Y, U, V1, V2 in a data frame
  data = data.frame(Y=Y, U=U, V1=V1, V2=V2) %>% as_tibble() 
  data
}

# Data basis manipulator
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

# Data generator part 2
makeKerasData = function(N) {
  dat0 = getData(N)
  dat1 = sincos(dat0["V1"] %>% rename(x=V1))
  dat2 = sincos(dat0["V2"] %>% rename(x=V2))
  dat3 = sincos(dat0["U"] %>% rename(x=U))
  dat = dat0 %>% select(Y) %>% bind_cols(dat1, dat2, dat3)
  dat
}

### Make train and test sets
dtest = makeKerasData(N)
xtest = dtest %>% select(-Y) %>% as.matrix()
ytest = dtest %>% select(Y) %>% as.matrix()

dtrain = makeKerasData(N*10)
xtrain = dtrain %>% select(-Y) %>% as.matrix()
ytrain = dtrain %>% select(Y) %>% as.matrix()

### Specify the architecture
model = keras_model_sequential() 
model %>%
  layer_dense(units = 20, activation = 'relu', input_shape = c(ncol(xtrain))) %>% 
  layer_dropout(rate = 0.5) %>%
  # layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'sigmoid') %>%
  layer_dense(units = 10, activation = 'linear') %>%
  layer_dense(units = 1, activation = 'linear')

summary(model)

### Specify loss and optimization method
model %>% compile(
  # loss = 'categorical_crossentropy',
  loss = c('mse'),
  optimizer = optimizer_nadam(clipnorm = 10),
  metrics = c('mse')
)

### Train model
early_stopping = callback_early_stopping(monitor = "val_loss", patience = 40)
bestLoss = 1e10
for(i in 1:20) {
  history = model %>% fit(xtrain, ytrain,
                          epochs = 50,
                          callbacks = c(early_stopping),
                          batch_size = 16,
                          validation_split = 0.2, shuffle=T
  )
  loss = history$metrics$val_loss[length(history$metrics$val_loss)]
  if(loss < bestLoss) {
    bestLoss = loss
    model %>% save_model_weights_hdf5("my_model_weights.h5")
  }
  if(length(history$metrics$val_loss) < 50)
    break
}

### Plot performance 
plot(history, metrics = "loss")  # only plots the last part of training

### Load the early-stopping model
bestModel = model %>% load_model_weights_hdf5('my_model_weights.h5')
bestModel %>% compile(
  # loss = 'categorical_crossentropy',
  loss = 'mse',
  optimizer = optimizer_nadam(),
  metrics = c('mse')
)

### Make predictions
bestModel %>% evaluate(xtest, ytest)
bestModel %>% predict_on_batch(xtest) %>% head()

# Compare against glmnet.
library(glmnet)
lasso = cv.glmnet(y = ytrain, x=xtrain, family = "gaussian")
preds = predict(lasso, newx = xtest)
sum((preds-ytest)^2)/length(preds)

### Your turn

### Part 1
### Modify the neural network to train faster
###   by using a hyperbolic tangent nonlinearity.
### Also, remove the dropout layer.
### Compare LASSO regression versus the model new model you created
###   in terms of mean squared error on the test set.

### Part 2
### Go to https://keras.rstudio.com/articles/examples/index.html
###   and find an example. Run the code. Discuss what the example does
###   with one to two people around you.