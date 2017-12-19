# This script builds a recurrent neural network (RNN) model
# for airline tweet data using the 'mxnet' package.

# LOAD LIBRARIES ----------------------------------------------------------
library(readr)
library(data.table)
library(mxnet)

# IMPORT DATA -------------------------------------------------------------
# Working directory must be set to the 'airline-tweets-nlp' repository folder.
data = data.table(read_csv('Airline-Sentiment-2-w-AA.csv'))

# TRANSFORM DATA ----------------------------------------------------------
# This section prepares the data for modeling.

  # Get only 'text' column from 'data'.
  # I'm only interested in the tweet text for the RNN.
  text = paste(data$text, 
               sep = ' ', 
               collapse = ' ')
  
  # Remove unwanted characters from 'text'.
  # Note that I couldn't remove forward slashes.
  text = gsub('@', '', text)  # '@' signs
  text = gsub('$', '', text)  #  dollar signs
  text = gsub(',', '', text)  # commas
  text = gsub(':', '', text)  # colons
  text = gsub(';', '', text)  # semi-colons
  text = gsub('-', '', text)  # dashes
  text = gsub('+', '', text)  # pluses
  text = gsub('#', '', text)  # number signs
  text = gsub('?', '', text)  # question marks
  text = gsub('!', '', text)  # exclamation marks
  text = gsub('\\.', '', text)  # periods
  text = gsub("'", "", text)  # apostrophes
  text = gsub('"', '', text)  # quotation marks
  text = gsub('/', '', text)  # backslashes
  text = gsub('&', '', text)  # '&' signs
  text = gsub('\\(', '', text)  # left parentheses
  text = gsub('\\)', '', text)  # right parentheses
  text = gsub('[[:digit:]]+', '', text)   # numbers
  
  # Make 'text' lowercase.
  text = tolower(text)
  
  text = gsub('virginamerica', '', text)  # airline tag
  text = gsub('americanair', '', text)  # airline tag
  text = gsub('united', '', text)  # airline tag
  text = gsub('southwestair', '', text)  # airline tag
  text = gsub('jetblue', '', text)  # airline tag
  text = gsub('usairways', '', text)  # airline tag
  text = trimws(text, 
                which = 'both')   # trim leading and trailing whitespace
  text = gsub('  ', ' ', text)  # replace double spaces with single spaces

  
  
  
  
  
    
# SET MODEL PARAMETERS ----------------------------------------------------

  # Set model parameters.
  # Will figure these out over time.
  batch.size = 32
  seq.len = 32
  num.hidden = 16
  num.embed = 16
  num.lstm.layer = 1
  num.round = 1
  learning.rate= 0.1
  wd=0.00001
  clip_gradient=1
  update.period = 1
  

# CREATE CHARACTER / ID DICTIONARIES ---------------------------------------------
# This section creates 2 dictionaries.
# The first is a dictionary where each dictionary name is a character and 
# the value is an ID.
# I use the lowercase letters, space, and a holder for other characters.
# The 2nd dictionary is the reverse: each dictionary name is an ID and each
# value is a character (for lowercase letters, space, and a holder for unknown characters).
# Characters are associated with the same ID in both dictionaries.
# For example, 'a' will be associated with 1 in both dictionaries.
  
  # Create word to ID dictionary.
  # The names of this dictionary are characters.
  
    # Create empty dictionary.
    word_id_dict = list()
    
    # Set initial dictionary ID (1).
    dict_id = 1
    
    # Loop through the lowercase letters. Add each one as a dictionary name.
    # The corresponding value is an increasing ID.
    for (letter in letters) 
      
    {
      
      # Add 'letter' to 'word_id_dict' with corresponding and increasing 'dict_id'.
      word_id_dict[[letter]] = dict_id
      
      # Increment 'dict_id' by 1.
      dict_id = dict_id + 1
      
    }
    
    # Add entry to 'word_id_dict' for space and increment 'dict_id'.
    word_id_dict[[' ']] = dict_id
    dict_id = dict_id + 1
    
    # Add entry to 'word_id_dict' for all other characters.
    word_id_dict[['other']] = dict_id
    
  # Create ID to word dictionary.
  # The names of this dictionary are numeric IDs.
    
    # Create empty dictionary.
    id_word_dict = list()
    
    # Loop through the names of 'word_id_dict' (characters).
    for character in names(word_id_dict)
    
      {
      
        #
        
      }

  