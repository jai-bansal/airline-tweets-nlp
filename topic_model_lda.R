# This script builds a topic model using latent Dirichlet allocation (LDA).

# LOAD LIBRARIES ----------------------------------------------------------
library(readr)
library(data.table)
library(dplyr)
library(ldatuning)
library(tm)
library(topicmodels)

# IMPORT DATA -------------------------------------------------------------
# Working directory must be set to the 'airline-tweets-nlp' repository folder.
data = data.table(read_csv('Airline-Sentiment-2-w-AA.csv'))

# CLEAN TEXT --------------------------------------------------------------
# This section removes unwanted symbols from the 'text' column.

  # Remove unwanted characters from 'data$text'.
  # Note that I couldn't remove forward slashes.
  data$text = gsub('@', '', data$text)  # '@' signs
  data$text = gsub('$', '', data$text)  #  dollar signs
  data$text = gsub(',', '', data$text)  # commas
  data$text = gsub(':', '', data$text)  # colons
  data$text = gsub(';', '', data$text)  # semi-colons
  data$text = gsub('-', '', data$text)  # dashes
  data$text = gsub('+', '', data$text)  # pluses
  data$text = gsub('#', '', data$text)  # number signs
  data$text = gsub('?', '', data$text)  # question marks
  data$text = gsub('!', '', data$text)  # exclamation marks
  data$text = gsub('\\.', '', data$text)  # periods
  data$text = gsub("'", "", data$text)  # apostrophes
  data$text = gsub('"', '', data$text)  # quotation marks
  data$text = gsub('/', '', data$text)  # backslashes
  data$text = gsub('&', '', data$text)  # '&' signs
  data$text = gsub('\\(', '', data$text)  # left parentheses
  data$text = gsub('\\)', '', data$text)  # right parentheses
  data$text = gsub('[[:digit:]]+', '', data$text)   # numbers
  
  # Make 'data$text' lowercase.
  data$text = tolower(data$text)
  
  data$text = gsub('virginamerica', '', data$text)  # airline tag
  data$text = gsub('americanair', '', data$text)  # airline tag
  data$text = gsub('united', '', data$text)  # airline tag
  data$text = gsub('southwestair', '', data$text)  # airline tag
  data$text = gsub('jetblue', '', data$text)  # airline tag
  data$text = gsub('usairways', '', data$text)  # airline tag
  data$text = trimws(data$text, 
                which = 'both')   # trim leading and trailing whitespace
  data$text = gsub('  ', ' ', data$text)  # replace double spaces with single spaces
  
  # Format the data for the (picky) 'tm' package.
  data = rename(data,                                   # rename a column
                doc_id = `_unit_id`)
  data = select(data, 
                c(doc_id, text, airline_sentiment, `airline_sentiment:confidence`, `negativereason`, `negativereason:confidence`, 
                  airline))
  
# REFORMAT DATA -----------------------------------------------------------
# This section formats the data so it's ready for LDA.
  
  # Create corpus.
  corp = Corpus(DataframeSource(data))
  
  # Create document-term-matrix (with lots of handy text cleaning included!).
  dtm = DocumentTermMatrix(corp, control = list(tolower = T, 
                                                removePunctuation = T, 
                                                removeNumbers = T, 
                                                stopwords = T, 
                                                stemming = T, 
                                                minWordLength = 2))
  
  # The above command apparently results in some rows (documents) with no words!
  # These need to be removed to proceed.
  
    # Compute the "sum" of words in each document (row).
    row_sum = apply(dtm, 1, sum)
    
    # Remove empty rows (where 'row_sum == 0') from 'dtm'.
    dtm = dtm[row_sum > 0, ]
  
# SELECT OPTIMAL NUMBER OF TOPICS -----------------------------------------
# This section selects the optimal number of topics for LDA.

  # Compute LDA performance for various numbers of topics and metrics.
  # I set the upper bound based on convenience and my own intuition.
  # This takes 2-4 minutes as structured below.
  topics_number_results = FindTopicsNumber(dtm, 
                                           topics = 2:10, 
                                           metrics = c('Griffiths2004', 'CaoJuan2009', 'Arun2010', 'Deveaud2014'),
                                           method = 'Gibbs', 
                                           control = list(seed = 20180116), 
                                           verbose = T)
  
  # View results.
  # I personally am not sure how helpful this is...
  # Arun2010 should be minimized but appears to decrease monotonically as number of topics increases
  # Deveaud2014 should be maximized and is maximized at 2 topics, after which it appears to decrease monotically 
  # in the number of topics.
  FindTopicsNumber_plot(topics_number_results)
  
  # Anyway, I'll just use 5 topics.
  topics = 5
  
# CONDUCT LDA AND ANALYZE ----------------------------------------------
# This section conducts the actual topic modeling using LDA and explores results.
  
  # Conduct LDA (this takes ~15 seconds).
  lda = LDA(x = dtm, 
            k = topics, 
            method = 'Gibbs')

  