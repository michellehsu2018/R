### use tidytext package (plus helpers) to characterize medical text

setwd("C:/Users/Jeremy/Box Sync/18 Spring/aa/lectures/obgyn_sample_cases/")
library(tidytext)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2); theme_set(theme_light())
library(wordcloud)

### Pre-processing ###
# load files into data frame df
df = data.frame(filenames = list.files(),stringsAsFactors = F) %>%
  tbl_df() %>%
  filter(str_detect(pattern=".txt", filenames)) %>%
  rowwise() %>%
  mutate(contents = file(filenames, "r") %>% readLines() %>% list())

# create data frame out of line list
df2 = df %>% mutate(formattedContents = 
                data.frame(line = 1:length(contents),
                           text = contents,
                           stringsAsFactors=F) %>% list()) %>% 
  select(-contents)

# tokenize into words
tokenized = df2 %>% 
  do(asTokens = .$formattedContents %>%
       unnest_tokens(word, text))

# remove stop words
tokenized = tokenized %>% mutate(noStops = asTokens %>% 
                                   anti_join(stop_words, by="word") %>% 
                                   list()
                                 )

### Analytics ###
# descriptive graphics
combinedTokens = tokenized %>% 
  do(counts = .$noStops %>% count(word, sort=T)) %>% ungroup() %>%
  unnest() %>% 
  group_by(word) %>% summarise(n=sum(n))
combinedTokens %>% 
  arrange(desc(n)) %>% head(20) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word,n)) + geom_col() + coord_flip()
combinedTokens %>%
  with(wordcloud(word, n, max.words=50))


# CONCEPT: TF-IDF term frequency-inverse document frequency:
# to see what each document is particularly focused on
tokenCounts = tokenized %>%
  do(countDfs = .$noStops %>% 
       count(word, sort=T) %>%
       left_join(combinedTokens %>% rename(totalN=n),by="word"))
# get count of documents per word
documentCounts = tokenCounts %>%
  do(hasWord = combinedTokens$word %in% 
       .$countDfs[["word"]] %>% as.numeric()) %>% 
  ungroup() %>% unnest() %>%
  (function(.) matrix(.[[1]],byrow = T, nrow=nrow(tokenCounts)))(.) %>%
  colSums() %>%
  data.frame(word = combinedTokens$word,
             nDocuments = .,
             stringsAsFactors = F) %>% tbl_df()
# combined tf and idf
tfidf = tokenCounts %>%
  do(word = .$countDfs[["word"]],
     tf = .$countDfs[["n"]]/max(.$countDfs[["n"]])
     ) %>%
  mutate(idf = match(word,documentCounts$word) %>%
           (function(.) documentCounts[.,"nDocuments"][[1]])(.) %>%
           list()
     ) %>%
  transmute(df = data.frame(word=word, 
                            tf=tf, 
                            idf=log(nrow(tokenCounts)/idf),
                            stringsAsFactors=F) %>%
              as_tibble() %>%
              mutate(tfidf = tf*idf) %>% 
              list())
# show
tfidf$df[[4]] %>% arrange(desc(tfidf)) %>% head(10)

# top tf-idf words from each document
tfidf %>% do(topWords = .$df[["word"]] %>% .[1:10]) %>% ungroup() %>%
  unnest() %>% t() %>% matrix(ncol=nrow(tfidf))


# correlation graph
topWords = combinedTokens %>% arrange(desc(n)) %>% head(n=30) %>% .[["word"]]
topCorrs = tokenCounts %>% 
  do(word = .$countDfs[["word"]]) %>%
  transmute(hasWord = topWords %in% word %>% list()) %>%
  unlist() %>% matrix(nrow=nrow(tokenCounts)) %>%
  cor()
library(igraph)
library(ggraph)
expand.grid(topWords, topWords, stringsAsFactors = F) %>%
  tbl_df() %>%
  (function(.) {names(.) = c("WordX","WordY"); .})(.) %>%
  mutate(corrs=topCorrs) %>% filter(corrs>0.5) %>%
  graph_from_data_frame() %>%
  ggraph(layout = "fr") +
  geom_edge_link(aes(edge_alpha = corrs), show.legend = FALSE) +
  geom_node_point(color = "lightblue", size = 5) +
  geom_node_text(aes(label = name), repel = TRUE) +
  theme_void()


# CONCEPT: LDA, latent dirichlet allocation
library(topicmodels)
# need a tibble with (document, word, n)
dtm = tokenCounts %>% 
  do(word = .$countDfs[["word"]],
     n = .$countDfs[["n"]]) %>%
  bind_cols(data.frame(document = df$filenames,stringsAsFactors=F)) %>%
  ungroup() %>%
  unnest() %>%
  cast_dtm(document, word, n)
topicmodel = LDA(dtm, k = 3, control = list(seed = 1234))
tidy(topicmodel, matrix="beta") %>% group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
# see the topics that the corpus gives

# Alternative pipeline: with vector embeddings
library(text2vec)
library(data.table)
prep_fun = tolower
tok_fun = word_tokenizer
data("movie_review"); mr = movie_review %>% tbl_df() %>% .[["review"]]
# mr = df$contents %>% unlist()
itmr = itoken(mr,
              preprocessor = prep_fun,
              tokenizer = tok_fun,
              chunks_number = 100)
# it = itoken(mr, preprocess_function = tolower,
#             tokenizer = word_tokenizer, chunks_number = 100)
vec = vocab_vectorizer(create_vocabulary(itmr)) #, skip_grams_window = 10)
tcm = create_tcm(itmr, vec)
glove = GlobalVectors$new(50, create_vocabulary(itmr), 10)
wvs = glove$fit_transform(tcm, n_iter = 20)
wv = glove$components
library(reshape2)
reorder_cormat = function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
qplot(x=Var1,
      y=Var2,
      data=wv[1:20,] %>%
        t() %>%
        cor() %>%
        reorder_cormat() %>%
        melt(), fill=value, geom="tile") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1,vjust = 0.25))
word_vectors = wvs + t(wv)  # main component + context
warmth = word_vectors["rain", , drop = FALSE] - 
  word_vectors["weather", , drop = FALSE] + 
  word_vectors["voice", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = warmth, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 5)

### Your turn: plot a word cloud using bi-grams ###
# Remove the stop words from the texts and recreate the bi-gram
# by modifying the code below.

tokenizedBigram = df2 %>%
  do(asTokens = .$formattedContents %>% 
       unnest_tokens(bigram, text, token="ngrams", n=2)
  )
tokenizedBigram %>%
  do(counts = .$asTokens %>% count(bigram, sort=T)) %>% 
  ungroup() %>%
  unnest() %>% 
  group_by(bigram) %>% summarise(n=sum(n)) %>% 
  with(wordcloud(bigram, n, max.words=50))
# this is why you don't use ngram wordclouds -- stop words problem!

# One solution
# tokenizedBigram %>% 
#   do(bigrams = .$asTokens %>% 
#        rowwise() %>% 
#        mutate(hasStop = any(str_split(bigram," ")[[1]] %in% 
#                               stop_words[["word"]])) %>% 
#        filter(!hasStop) %>% 
#        select(-hasStop)) %>%
#   unnest() %>% ungroup() %>% 
#   group_by(bigram) %>% count() %>% arrange(desc(n)) %>%
#   with(wordcloud(bigram, n, max.words=10, min.freq=2, scale = c(2,0.8)))
  


# see http://tidytextmining.com/topicmodeling.html#latent-dirichlet-allocation
# for much more you can show from text analytics.
# The Language Technologies Institute at CMU goes
# into much greater detail.