library(dplyr); library(purrr); library(tidyr)

# Toy data for performing linear regression in 5-fold CV
N = 23
mydata = 
  data.frame(id=1:N) %>% as_tibble() %>%
  mutate(x = rnorm(dim(.)[1])) %>% #[1]:indicates rows number #(.)takes whatever gets pipe in, in this case is a data frame
  mutate(y = 2*x + rnorm(dim(.)[1]))


prep =  # two columns
  data.frame(id=1:N) %>% as_tibble() %>%
  mutate(testpart=rep(1:5, length.out=dim(.)[1]))#loop 1 to 5 until all the rows are filled 

#tydyr
nestedprep =  # make a list column
  prep %>% nest(-testpart)       

nestedprep[[1,2]] #list column within a data frame: index use [[]] ==> dereference object!

#purrr
withtest =  # now start adding columns based on test (or train) set ids in 'data'
  nestedprep %>%
  mutate(testdata = map(data, ~ mydata %>% filter(id %in% .x$id))) 

withall = 
  withtest %>%
  mutate(traindata = map(data, ~ mydata %>% filter(!id %in% .x$id)))

withlm =
  withall %>%
  mutate(linear = map(traindata, ~ lm(y~x, data = .x))) #once for every row; train on the training data

withprediction = 
  withlm %>%
  dplyr::select(-data) %>%  # clean up a little
  mutate(pred = map2(linear, testdata, ~ predict(.x, .y)))# .x first argument; .y second argument

withtruth = 
  withprediction %>%
  mutate(truth = map(testdata, ~ .x["y"] %>% t() %>% c())) %>%
  dplyr::select(-traindata, -testdata)

# Get out of list columns
withtruth %>% select(pred, truth) %>% unnest()  # or
data.frame(preds = withtruth$pred %>% unlist(),
           truths = withtruth$truth %>% unlist()) %>% as_tibble()

#caliberated porbability: the actual probability not the ranking

# All together:
data.frame(id=1:19) %>% as_tibble() %>%
  mutate(testpart=rep(1:5, length.out=dim(.)[1])) %>%
  nest(-testpart) %>%
  mutate(testdata = map(data, ~ mydata %>% filter(id %in% .x$id))) %>%
  mutate(traindata = map(data, ~ mydata %>% filter(!id %in% .x$id))) %>%
  mutate(linear = map(traindata, ~ lm(y~x, data = .x))) %>%
  mutate(pred = map2(linear, testdata, ~ predict(.x, .y))) %>%
  mutate(truth = map(testdata, ~ .x["y"] %>% t() %>% c())) #transpose and concatenant==>transform tibble into a vector


### Your turn ###
# Take the data.frame prep and nest it leaving testpart outside the list column
# Use mutate(mycolumnname = map_lgl(data , ...)) to test if id 5 is in the list column
#   note use of map_lgl instead of map() because we want a logical return value
prep %>% as_tibble() %>%
  nest(-testpart) %>%
  mutate(flag = map_lgl(data, ~ 5 %in% .x$id))

