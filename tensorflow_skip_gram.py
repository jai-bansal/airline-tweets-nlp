# This script builds a skip-gram model for airline tweet data
# using the 'tensorflow' package.

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import os
import pandas as pd
import tensorflow as tf
import collections as col # 'collections' module deals with data containers and counting. 

#############
# IMPORT DATA
#############
# This section imports data.
# Working directory must be set to the 'airline-tweets-nlp' repository folder.
data = pd.read_csv('Airline-Sentiment-2-w-AA.csv',
                   encoding = 'latin-1')

################
# TRANSFORM DATA
################
# This section prepares the data for modeling.

# Get lowercase text data.
# I'm only interested in the tweet texts.
text = data.text.str.lower()

# Remove unwanted characters from 'text'.
# Note that I couldn't remove forward slashes.
text.replace(regex = True, inplace = True, to_replace = '@', value = '')  # '@' signs
text.replace(regex = True, inplace = True, to_replace = '\$', value = '')  # '$' signs
text.replace(regex = True, inplace = True, to_replace = ',', value = '')  # commas
text.replace(regex = True, inplace = True, to_replace = ':', value = '')  # colons
text.replace(regex = True, inplace = True, to_replace = ';', value = '')  # semi-colons
text.replace(regex = True, inplace = True, to_replace = '-', value = '')  # minus signs
text.replace(regex = True, inplace = True, to_replace = '\+', value = '')  # plus signs
text.replace(regex = True, inplace = True, to_replace = '#', value = '')  # hashtags
text.replace(regex = True, inplace = True, to_replace = '\?', value = '')  # question marks
text.replace(regex = True, inplace = True, to_replace = '\!', value = '')  # question marks
text.replace(regex = True, inplace = True, to_replace = '\.', value = '')  # periods
text.replace(regex = True, inplace = True, to_replace = '\'', value = '')  # apostrophes
text.replace(regex = True, inplace = True, to_replace = '\"', value = '')  # quotes
text.replace(regex = True, inplace = True, to_replace = '\/', value = '')  # backslashes
text.replace(regex = True, inplace = True, to_replace = '\&', value = '')  # '&' signs
text.replace(regex = True, inplace = True, to_replace = '\(', value = '')  # parentheses
text.replace(regex = True, inplace = True, to_replace = '\)', value = '')  # parentheses
text.replace(regex = True, inplace = True, to_replace = '\d+', value = '')  # all digits
text.replace(regex = True, inplace = True, to_replace = 'virginamerica', value = '')  # airline tag
text.replace(regex = True, inplace = True, to_replace = 'americanair', value = '')  # airline tag
text = text.str.strip() # remove leading and trailing spaces
text.replace(regex = True, inplace = True, to_replace = '  ', value = ' ')  # double spaces with spaces

# Turn 'text' into 1 large string.
# These are separate tweets being combined, instead of a coherent document, but oh well.
single_string = text.str.cat(sep = ' ')

# Convert to 'tensorflow' string and split on spaces.
tf_string = tf.compat.as_str(single_string).split()

##################
# BUILD DICTIONARY
##################
# This section builds the dictionary and replaces rare words with an 'unknown' token.

# Choose size of vocabulary for dictionary.
vocab_size = 5000

# Create function to get all necessary objects from 'tf_string' to create model.
def build_dataset(string):

    # Create list with 1 initial entry.
    counts = [['unknown', -1]]

    # Get counts of the most common words.
    # '.extend' extends 'counts'.
    # 'collections.Counter' creates a dictionary with words and counts of those words.
    # '.most_common' limits the size of the dictionary.
    counts.extend(col.Counter(string).most_common(vocab_size - 1))

    # Create dictionary that contains each word in 'counts' as key
    # and gives it an index for the value.
    word_dict = dict()
    for word, _ in counts:
        word_dict[word] = len(word_dict)

    # Create list to hold the index of each word in 'string' in order.
    string_indices = list()

    # Create counter for unknown words.
    unknown_counter = 0

    # Fill in 'string_indices'.
    for word in string:

        if word in word_dict:

            # For words in 'word_dict', get index. 
            index = word_dict[word]

        else:

            # For words not in 'word_dict' set index to 0, which corresponds to 'unknown'.
            index = 0
            
            # Increment unknown word counter ('unknown_counter') by 1.
            unknown_counter += 1

        # Add 'index' to 'string_indices'.
        string_indices.append(index)

    # 'counts[0]' is still '['unknown', -1]'.
    # Replace the '-1' with 'unknown_counter', the true count of unknown words.
    counts[0][1]  = unknown_counter

    # Create reverse dictionary.
    # So, the keys are the indices from 'word_dict' and the values are the words.
    reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))

    # Return function outputs.
    return(string_indices, counts, word_dict, reverse_word_dict)

# Call 'build_dataset' on 'tf_string'.
string_indices, counts, word_dict, reverse_word_dict = build_dataset(tf_string)

################################################################
# CREATE FUNCTION TO GENERATE TRAINING BATCH FOR SKIP-GRAM MODEL
################################################################
# This section creates a function that generates a training
# batch for the skip-gram model.

# Create index variable to track position in 'string_indices'.
string_index = 0

# Define function to generate training batch for skip-gram model.
# 'batch_size
def generate_training_batch(batch_size, samples, sample_range):























