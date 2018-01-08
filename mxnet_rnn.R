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
# Enter '?mx.lstm' to find the descriptions of parameters used below.
# Strangely, I can't find the help file online...

  # Set model parameters
  batch_size = 32             # batch size
  batch_length = 16           # length of each string in input (in characters)
  nodes = 32                  # number of hidden nodes
                              # I think this corresponds to 'nodes' in the Python RNN script
  embedding_size = 64         # the output dim of embedding
                              # I think this corresponds to 'embedding_size' in the Python RNN script
                              # So, the number of columns in the embedding matrix
  num_lstm_layer = 1          # From documentation: the number of the layer of LSTM
                              # I guess you could have multiple LSTM cell layers?
                              # In that case, this parameter would control how many?
                              # Not too clear on what this parameter does...
  num_round = 2               # number of iterations of the training data to train the model
  learning_rate = 0.1         # Assumption: set learning rate for optimizer (same as 'learning_rate' in Python RNN).
                              # Note that the learning rate in the Python RNN was decaying.
  char_to_generate = 80       # how many characters to generate when projecting forward with the model

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
    char_id_dict = list()
    
    # Set initial dictionary ID (1).
    dict_id = 1
    
    # Loop through the lowercase letters. Add each one as a dictionary name.
    # The corresponding value is an increasing ID.
    for (letter in letters) 
      
      {
        
        # Add 'letter' to 'char_id_dict' with corresponding and increasing 'dict_id'.
        char_id_dict[[letter]] = dict_id
        
        # Increment 'dict_id' by 1.
        dict_id = dict_id + 1
        
      }
    
    # Add entry to 'char_id_dict' for space and increment 'dict_id'.
    char_id_dict[[' ']] = dict_id
    dict_id = dict_id + 1
    
    # Add entry to 'char_id_dict' for all other characters.
    char_id_dict[['other']] = dict_id
    
  # Create ID to word dictionary.
  # The names of this dictionary are numeric IDs.
    
    # Create empty dictionary.
    id_char_dict = list()
    
    # Loop through the names of 'char_id_dict' (characters).
    for (char in names(char_id_dict))
    
      {
      
        # Get the value corresponding to the name 'char' in 'char_id_dict'.
        id = char_id_dict[[char]]
        
        # Create an element in 'id_char_dict' with name 'id' and value 'char'.
        id_char_dict[[id]] = char
        
      }

# CREATE FUNCTION TO SAMPLE FROM PROBABILITY DISTRIBUTION -----------------
# This section creates a function to sample the index from a probability distribution.
# So for the probability distribution (0.3, 0.2, 0.5), the function
# would return 1 30% of the time, 2 20% of the time, and 3 50% of the time.
    
  # Create helper function to turn a probability distribution into a cumulative
  # distribution function.
  pdf_to_cdf = function(prob_dist) 
    
    {
    
      # Sum all probabilities in 'prob_dist'.
      # I guess this function works for non-normalized probability distributions that don't add up to 1.
      total_prob = sum(prob_dist)
      
      # Create empty vector for resulting CDF.
      cdf = c()
      
      # Create variable that will hold each value of the CDF (one at a time).
      cdf_element = 0
      
      # Iterate through probabilities in 'prob_dist'.
      # Add the probability distribution element to 'cdf_element'.
      # Divide that value by 'total_prob' and add to 'cdf'.
      for (prob in prob_dist) 
        
        {
        
          cdf_element = cdf_element + prob
          cdf = c(cdf, cdf_element / total_prob)
        
        }
      
      return(cdf)
    
    }
  
  # Create helper function that, given a value of 'x' between 0 and 1, 
  # returns the index of the smallest element of 'cdf' such that 'cdf >= x'.
  # This function is pretty unintuitive on its own, but is used below 
  # to help randomly sample an index from a probability distribution.
  search = function(cdf, cutoff) 
    
    {
    
      smallest_index = 1
      r = length(cdf)
  
      # While 'smallest_index <= r', depending on the relationship between 'cdf[m]' and cutoff, 
      # either add to 'smallest_index' or decrease from 'r'.
      while (smallest_index <= r) 
        
        {
        
          m = as.integer((smallest_index + r) / 2)
    
          if (cdf[m] < cutoff) {smallest_index = m + 1} 
          
          else {r = m - 1}  
        
        }
      
      return(smallest_index)
      
    }
    
  # Create function to randomly sample the index from a probability distribution 
  # according to the probabilities in the distribution.
  # This function is the main output of this section.
  # So for the probability distribution (0.3, 0.2, 0.5), the function below
  # returns 1 30% of the time, 2 20% of the time, and 3 50% of the time.
  sample_index = function(prob_dist) 
    
    {
    
      # Compute the cumulative distribution function of 'prob_dist' using 'pdf_to_cdf' above.
      cdf = pdf_to_cdf(as.array(prob_dist))
      
      # Randomly pick a value from the uniform distribution (0, 1).
      random_value = runif(1)
      
      # Run 'search' function from above using the CDF just generated as the 'cdf' argument 
      # and 'random_value' as cutoff. This results in the desired function behavior.
      index = search(cdf, random_value)
      
      return(index)
      
    }
    
# PUT TEXT DATA INTO NUMERIC ARRAY ----------------------------------------
# This section puts the text data into a numeric array.
# The number of rows is the length of each string in a batch.
# The number of columns is the number of strings that can fit into the text.
# That value can be computed based on the length of each string and the 
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
    # I will loop through every character in 'text'.
    text_index = 1
  
    # Loop through each string.
    for (string in 1:total_strings)
      
      {
      
        # Print progress.
        if (string %% 30000 == 0) {print(string)}
        
        # Loop through each character of each string.
        for (char in 1:batch_length)
          
          {
          
            # If the 'text_index' character of 'text' is in 'char_id_dict', 
            # change the corresponding entry in 'text_array' to the corresponding 
            # 'char_id_dict' value (a number).
            # Subtracting 1 below means the values in 'text_array' run from 0 to 27, 
            # instead of 1 to 28. This is apparently key, as the model returns a lot 
            # of 'Inf' and 'NaN' values without this step.
            # Why? Not sure. Maybe because it relies on a Python implementation and Python is zero-indexed??
            if (text[text_index] %in% names(char_id_dict))
              
              {
                
                # Update corresponding entry of 'text_array'.
                text_array[char, string] = char_id_dict[[text[text_index]]] - 1
              
              }
          
            # If the 'text_index' character of 'text' is NOT in 'char_id_dict', 
            # change the corresponding entry in 'text_array' to the value 
            # corresponding to 'other' in 'char_id_dict'.
            # Subtracting 1 below means the values in 'text_array' run from 0 to 27, 
            # instead of 1 to 28. This is apparently key, as the model returns a lot 
            # of 'Inf' and 'NaN' values without this step.
            # Why? Not sure. Maybe because it relies on a Python implementation and Python is zero-indexed??
            else 
              
              {
              
                # Update corresponding entry of 'text_array'.
                text_array[char, string] = char_id_dict[['other']] - 1
                
              }
          
            # Increment 'text_index'.
            text_index = text_index + 1
          
          }
      
      }

  # Make 'text_array' such that it is divisible by 'batch_size' with no remainder.
  # This informs the number of steps the model can run without repeat.
  
    # Compute the number of batches contained in 'text'.
    batches = floor(ncol(text_array) / batch_size)
    
    # Trim 'text_array' so it contains a number of columns (batches) that can 
    # be divided by 'batch_size' with no remainder.
    text_array = text_array[, 1:(batch_size * batches)]
    
# GET LABELS --------------------------------------------------------------
# This section gets labels corresponding to the values in 'text_array'.
# The labels are generally the same values as 'text_array' shifted by 1 position.
    
  # Get variables for the rows and columns in 'text_array'.
  rows = dim(text_array)[1]
  cols = dim(text_array)[2]
    
  # Create array to hold labels.
  labels = array(0, 
                 dim = dim(text_array))
    
  # Fill in 'labels'.
  # Loop through columns.
  for (col in 0:(cols - 1))
    
    {
      
      # Keep track of loop progress.
      if (col %% 20000 == 0) {print(col)}
    
      # Loop through rows.
      for (row in 1:rows)
        
        {
          
          # Fill in 'labels'.
          # The value of 'labels' is the corresponding position in 'text_array' shifted by 1 position.
          labels[(col * rows) + row] = text_array[((col * rows) + row) %% (rows * cols)  + 1] 
          
        }
      
    }
    
# CREATE TRAINING AND VALIDATION SETS -------------------------------------
# This section creates the training and validation sets.
# I use roughly an 80/20 training/validation split.
# It's not exactly 80/20 because I make sure both the training and validation sets
# have exactly enough characters to complete a batch.
  
  # Get training and validation set input data.
  train_data = text_array[, 1 : floor(0.8 * ncol(text_array))]
  val_data = text_array[, ceiling(0.8 * ncol(text_array)) : ncol(text_array)]
  
  # Get training and validation set labels.
  train_labels = labels[, 1 : floor(0.8 * ncol(text_array))]
  val_labels = labels[, ceiling(0.8 * ncol(text_array)) : ncol(text_array)]
  
  # Combine input data and labels in lists for training and validation data.
  train_list = list(data = train_data, 
                    labels = train_labels)
  val_list = list(data = val_data, 
                  labels = val_labels)
  
# TRAIN MODEL -------------------------------------------------------------
# This section trains the LSTM.
# This seems very fragile...changing parameters can make the model return nonsense.
# Removing arguments that are specified by default can make the model return nonsense.
# NLL seems like some kind of error as it decreases.
# I assume 'Perp' is perplexity.
model = mx.lstm(train_list, val_list,
                 num.round = num_round,
                 num.lstm.layer = num_lstm_layer,
                 seq.len = batch_length,
                 num.hidden = nodes,
                 num.embed = embedding_size,
                 num.label = length(char_id_dict),
                 batch.size = batch_size,
                 input.size = length(char_id_dict),
                 initializer = mx.init.uniform(0.1),
                 learning.rate = learning_rate)
    
# GENERATE NEW TEXT BASED ON LSTM -----------------------------------------
# This section generates new text based on the LSTM.

  # Build inference from the model.
  # 'mxnet' is high level...this just seems like magic.
  model_inference = mx.lstm.inference(num.lstm.layer = num_lstm_layer,
                                      input.size = length(char_id_dict),
                                      num.hidden = nodes,
                                      num.embed = embedding_size,
                                      num.label = length(char_id_dict),
                                      arg.params = model$arg.params)
  
  # Pick random starting character and get numeric ID (but not 'other' character).
  current_char_id = sample(1:(length(char_id_dict) - 1), 1)
  
  # Get actual starting character and start generated sentence.
  generated_sentence = id_char_dict[[current_char_id]]
  
  # I have 1 character. Loop through to generate the rest.
  for (i in 1:(char_to_generate - 1))
    
  {
    
    # Predict next character using 'model_inference'.
    # I use 'current_char_id - 1' because I trained the model this way.
    # See the "PUT TEXT DATA INTO NUMERIC ARRAY" section for more details.
    forward_pred = mx.lstm.forward(model_inference, 
                                   (current_char_id - 1))
    
    # Update 'model_inference'.
    model_inference = forward_pred$model
    
    # One of the outputs of 'forward_pred' is a probability distribution over the next character.
    # Sample this probability distribution using the 'sample_index' function above.
    # Update 'current_char_id' with this sampled value.
    current_char_id = sample_index(forward_pred$prob)
    
    # Add the corresponding character for 'current_char_id' to 'generated_sentence'.
    generated_sentence = paste0(generated_sentence, id_char_dict[[current_char_id]])
    
  }
  
  print(generated_sentence)