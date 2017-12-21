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
  batch.size = 32             # guess: batch size
  
  seq.len = 32                # educated guess: length of each string in a batch
  # Set length of each string in the batch (in characters).
  batch_length = 10
  
  num.hidden = 16             # guess: # of hidden nodes
  num.embed = 16
  num.lstm.layer = 1
  num.round = 1
  learning.rate= 0.1          # guess: RNN learning rate
  wd = 0.00001
  clip_gradient=1
  update.period = 1
  

# CREATE CHARACTER / ID DICTIONARIES ---------------------------------------------
# This section creates 2 dictionaries.
# In the first, each dictionary name is a character and the value is a numeric ID.
# I use the lowercase letters, space, and a holder for other characters.
# The 2nd dictionary is the reverse: each dictionary name is a numeric ID and each
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
    for (char in names(word_id_dict))
    
      {
      
        # Get the value corresponding to the name 'char' in 'word_id_dict'.
        id = word_id_dict[[char]]
        
        # Create an element in 'id_word_dict' with name 'id' and value 'char'.
        id_word_dict[[id]] = char
        
      }

    


# PUT TEXT DATA INTO NUMERIC ARRAY ----------------------------------------
# This section puts the text data into a numeric array.
# The number of rows is the length of each string in a batch.
# The number of columns is the number of strings that can fit into the text.
# The value can be computed based on the length of each string and the 
# length of the text.

  # Split 'text' by character.
  text = strsplit(text, '')[[1]]
    
  # Compute the number of full strings that can fit into 'text'.
  # String length is defined by 'batch' length.
  total_strings = floor(length(text) / batch_length)
  
  # Only keep enough characters of 'text' to complete 'total_strings' 
  # strings of 'batch_length' length.
  text = text[1:(total_strings* batch_length)]
  
  # Create empty array.
  text_array = array(0, dim = c(batch_length, total_strings))
  
  # Fill in 'text_array' with the appropriate integer for each character.
  
    # Set initial text index to 1.
    # I will loop through every character of 'text'.
    text_index = 1
  
    # Loop through each string.
    for (string in 1:total_strings)
      
      {
      
        # Print progress.
        if (string %% 30000 == 0)
          
          {
          
            print(string)
          
          }
        
        # Loop each character of each string.
        for (char in 1:batch_length)
          
          {
          
            # If the 'text_index' character of 'text' is in 'word_id_dict', 
            # change the corresponding entry in 'text_array' to the corresponding 
            # 'word_id_dict' value.
            if (text[text_index] %in% names(word_id_dict))
              
              {
                
                # Update corresponding entry of 'text_array'.
                text_array[char, string] = word_id_dict[[text[text_index]]]
              
              }
          
            # If the 'text_index' character of 'text is NOT in 'word_id_dict', 
            # change the corresponding entry in 'text_array' to the value 
            # corresponding to 'other' in 'word_id_dict'.
            else 
              
              {
              
                # Update corresponding entry of 'text_array'.
                text_array[char, string] = word_id_dict[['other']]
                
              }
          
            # Increment 'text_index'.
            text_index = text_index + 1
          
          }
      
      }

    