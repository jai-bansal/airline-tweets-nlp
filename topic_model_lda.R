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
    
  # Now, I want to see if I want to keep all of the words.
  # The criteria below are very heuristic and are based somewhat on this video:
  # https://www.youtube.com/watch?v=3mHy4OSyRf0
  
    # First, I'll count how many times each word occurs. There are 14623 documents.
    # Note that a word can occur more than once in a document.
    word_occurrences = apply(dtm, 2, sum)
    
    # Remove words that appear once (and thus only in one document).
    dtm = dtm[1:nrow(dtm), word_occurrences > 1]
    
    # Now, I'll count how many DOCUMENTS a word appears in.
    doc_occurrences = apply(dtm, 2, function(x) {sum(x != 0)})
    
    # The most occurring word appears 4817 times in 3908 documents. It's "flight".
    
    # Remove words that appear in more than 5% of articles. This is one of those heuristic metrics.
    # The video referenced above removes words appearing in more than 10% of articles, but I don't have
    # that much data so...5% it is.
    dtm = dtm[1:nrow(dtm), doc_occurrences < as.integer(nrow(dtm) / 20)]
    
  # The above work apparently results in some rows (documents) with no words!
  # These need to be removed to proceed.
    
    # Compute the "sum" of words in each document (row).
    row_sum = apply(dtm, 1, sum)
    
    # Remove empty rows (where 'row_sum == 0') from 'dtm'.
    dtm = dtm[row_sum > 0, ]
    
  # Update 'data' to reflect the removed documents.
    
    # Save a copy.
    orig_data = data
    
    # Update 'data'.
    data = data[row_sum > 0]
  
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
  # Griffiths 2004 should be maximized but appears to increase monotonically as the number of topics increases
  FindTopicsNumber_plot(topics_number_results)
  
  # Anyway, I'll just use 5 topics.
  topics = 5
  
# CONDUCT LDA AND ANALYZE ----------------------------------------------
# This section conducts the actual topic modeling using LDA and explores results.
  
  # Conduct LDA (this takes ~15 seconds).
  lda = LDA(x = dtm, 
            k = topics, 
            method = 'Gibbs')
  
  # View the "top" 20 terms in each topic.
  terms(lda, 20)
  
  # View the breakdown of how many documents fall into each topic.
  table(topics(lda))
  
  # View probabilities of belonging to each topics per document.
  head(posterior(lda)$topics)
  
  # Add topic ID to original 'data'.
  data$topic_id = topics(lda)
  
  # I can now look for correlations between the derived topics and some of the metadata.
  
    # Compare derived topic to 'airline_sentiment'.
    # Most tweets were negative, but...
    # Topic 4 trends very negative.
    # Topic 5 seems to be slightly less negative.
    table(data$topic_id, data$airline_sentiment)
    
    # Compare derived topic to 'negativereason' where it exists.
    # Topic 1 got the majority of 'Damaged Luggage' and 'Lost Luggage'
    # Topic 2 got the majority of 'Flight Booking Problems'
    # Topic 3 got the majority of 'Late Flight'
    # Topic 4 got the majority of 'Customer Service Issue'
    table(data$topic_id, data$negativereason)
  