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
print(text[100])

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





