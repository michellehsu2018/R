#install.packages("devtools")
#devtools::install_github("rstudio/keras")
library(keras)
library(dplyr); library(readr)

N = 500
K = 50

### Get data generator, train and test
getData = function(N=1000, K = 100) {
  x=matrix(0, N, N)
  for (i in 1:100) {
    xlb = sample(N,1)
    xub = min(N, xlb + sample(K,1))
    ylb = sample(N,1)
    yub = min(N, ylb + sample(K,1))
    x[xlb:xub, ylb:yub] = rpois(lambda = 20, (xub-xlb+1)*(yub-ylb+1))
  }
  x
}

input_size = 64
noise = matrix(rnorm(N*64), N, input_size)  # each row is an input
dat = getData(N,K)  # each row will be considered an output
hidden_size = 64

### Show dat and noise
image(dat, useRaster=TRUE, axes=FALSE, col=terrain.colors(12))
image(noise, useRaster=TRUE, axes=FALSE, col=terrain.colors(12))


### Specify the architecture
model = keras_model_sequential() 
model %>%
  layer_dense(units = hidden_size, activation = 'elu',
              input_shape = c(ncol(noise))) %>%
  layer_dense(units = hidden_size, activation = 'tanh') %>%
  layer_dense(units = hidden_size, activation = 'tanh') %>%
  layer_dense(units = hidden_size,kernel_regularizer = regularizer_l2(l = 0.01)) %>%
  layer_activation_leaky_relu(alpha=0.1) %>%
  layer_dense(units = dim(dat)[2],
              activation = 'elu')

summary(model)
#LeakyReLU: prevent the negative side of weight to become 0 in case you lose a lot of information and not update any node.
#dense: nodes are connected

### Specify loss, batch size, optimizer, extra performance measures
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_nadam(clipnorm = 10),
  metrics = c('accuracy')
)


### Run model to learn weights
for (i in 1:10) {
  print(paste("round", i))
  history = model %>% fit(noise, dat, 
                          epochs = 50,
                          batch_size = 64,
                          verbose = F
  )
}
#fitting 500 times

model %>% evaluate(noise, dat)
# model %>% get_weights()
plot(history)

par(mfrow=c(1,2))
image(dat, useRaster=TRUE, axes=FALSE, col=terrain.colors(12))
image(model %>% predict(noise),
      useRaster=TRUE, axes=FALSE, col=terrain.colors(12))
#the color is slightly different due to regularization

image(dat, useRaster=TRUE, axes=FALSE, col=terrain.colors(12))
image(model %>% predict(noise + matrix(0.3*rnorm(N*input_size),N,input_size)), #adding noise
      useRaster=TRUE, axes=FALSE, col=terrain.colors(12))

image(dat, useRaster=TRUE, axes=FALSE, col=terrain.colors(12))
image(model %>% predict(noise + matrix(1.0*rnorm(N*input_size),N,input_size)), #adding noise
      useRaster=TRUE, axes=FALSE, col=terrain.colors(12))

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