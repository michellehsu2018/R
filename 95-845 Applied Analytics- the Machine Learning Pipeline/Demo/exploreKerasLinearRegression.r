#install.packages("devtools")
#devtools::install_github("rstudio/keras")
library(keras)
#install_keras()
#Neural Network loses interpretability but generate strong prediction
library(dplyr); library(readr)

N = 500

### Get data generator, train and test
getData = function(N=100) {
  data = data.frame(x=matrix(rnorm(N*5), N, 5)) %>% as_tibble() %>%
    mutate(Y = x.1 + 3*x.2 + 10*x.3 + x.4*x.4 + x.5*x.5*x.5) #x.3 is the column 3 in terms of matrix
  data
}

dtest = getData(N)
xtest = dtest %>% select(-Y) %>% as.matrix()
ytest = dtest %>% select(Y) %>% as.matrix()

dtrain = getData(N*10)
xtrain = dtrain %>% select(-Y) %>% as.matrix()
ytrain = dtrain %>% select(Y) %>% as.matrix()


### Specify the architecture
model = keras_model_sequential() 
model %>%
  layer_dense(units = 20, activation = 'relu', input_shape = c(ncol(xtrain))) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'sigmoid') %>%
  layer_dense(units = 10, activation = 'linear') %>%
  layer_dense(units = 1, #predict 1 y, 1 output
              activation = 'linear',
              input_shape = ncol(xtrain))

summary(model)

### Specify loss, batch size, optimizer, extra performance measures
model %>% compile(
  # loss = 'categorical_crossentropy',
  loss = c('mse'),
  optimizer = optimizer_nadam(clipnorm = 10), #some other type of gradient descent instead of scohastic gradient descent
  metrics = c('mse')
)


### Run model to learn weights
history = 
  model %>% fit(xtrain, ytrain,
              epochs = 20, #number of trips to your database
              batch_size = 16,
              validation_split = 0.2, shuffle=T
  )

model %>% evaluate(xtest, ytest)
model %>% get_weights()
plot(history)

### Your turn
# Part 1. 
# Increase the depth of the neural network. You will have to change the code
# so that the first layer has the argument "input_shape".
# Make sure to include a non-linearity somewhere (i.e. activation function),
# because a linear combination of linear combinations is just a single
# linear combination.

### Part 2
# Go to https://keras.rstudio.com/articles/examples/index.html
#   and find an example. Run the code. Discuss what the example does
#   with one to two people around you.
