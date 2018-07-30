library(dplyr); library(tidyr); library(purrr); library(readr)

dat = read_csv("../../Dropbox/18 Spring/toydiabetesdata.csv")
dat %>% nest(-pt)

dat %>% dplyr::select(pt) %>% unique() %>% 
  crossing(data.frame(x=1:(dat$pt %>% unique() %>% length()))) %>% 
  filter(pt!=x) %>% nest(-pt) %>%  # create the LOOCV #
  mutate(data = map(data, ~ .x %>% t() %>% c())) %>%  # make the tibble "data" something easily indexable (a vector)
  mutate(train = map(data, ~ dat %>% filter(pt %in% .x))) %>%  # collect the data (or pipe it to a learner immediately to be more memory efficient)
  mutate(test = map(data, ~ dat %>% filter(!pt %in% .x)))


### Your turn ###
# Create a train tune test split on pts (50/25/25) when samples are given in long format or other (i.e. aren't a fixed length feature vector)
# Goal: generate a tibble with set = c("Train","Tune","Test") and a list column of vectors of pts in each
# - step 1. make two columns or arrays: the unique pt ids, and the labels "Train", "Tune", "Test" in the appropriate number
# - step 2. permute one of them.
# - step 3. make a tibble
# - step 4. nest the tibble
# - step 5. use the ptids in the list column to filter the data using mutate(dataset = map(ptids, ~ ...))
# Bonus: create cross-validation train-tune splits

ptids = dat$pt %>% unique()
mlset = c(rep("Train",length(ptids)/2),
          rep("Tune", length(ptids)/4),
          rep("Test", length(ptids)/4))
mldata = data.frame(pt = ptids, set = sample(mlset)) %>% as_tibble() %>% nest(-set)
mldata %>% mutate(dataset = map(data, ~ dat %>% filter(pt %in% .x)))
