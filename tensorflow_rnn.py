# This script builds a recurrent neural network (RNN) model
# for airline tweet data using the 'tensorflow' package.

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import pandas as pd
import string
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
text.replace(regex = True, inplace = True, to_replace = 'united', value = '')  # airline tag
text.replace(regex = True, inplace = True, to_replace = 'southwestair', value = '')  # airline tag
text.replace(regex = True, inplace = True, to_replace = 'jetblue', value = '')  # airline tag
text.replace(regex = True, inplace = True, to_replace = 'usairways', value = '')  # airline tag
text = text.str.strip() # remove leading and trailing spaces
text.replace(regex = True, inplace = True, to_replace = '  ', value = ' ')  # double spaces with spaces

# Turn 'text' into 1 large string.
# These are separate tweets being combined, instead of a coherent document, but oh well.
single_string = text.str.cat(sep = ' ')

# Convert to 'tensorflow' string and split on spaces.
tf_string = tf.compat.as_str(single_string)

#######################
# CREATE VALIDATION SET
#######################

# Choose size of validation set.
valid_size = 1000

# Divide 'tf_string' into training and validation sets.
valid_text = tf_string[:valid_size]
train_text = tf_string[valid_size:]

############################################
# CREATE CHARACTER-NUMERIC MAPPING FUNCTIONS
############################################
# This section creates functions that map characters to numeric IDs and vice versa.

# Get the Unicode integer for 'a'.
a_int = ord(string.ascii_lowercase[0])

# Create function to map characters to numeric IDs.
# 'a' maps to 1, 'z' maps to 26. The middle is obvious.
# Spaces and unexpected / unrecognized characters map to 0.
def char_to_num(character):

    # Map valid characters to relevant numeric IDs.
    if character in string.ascii_lowercase:
        return(ord(character) - a_int + 1)

    # Map spaces to numeric ID 0.
    elif character == ' ':
        return(0)

    # Map unexpected / unrecognized characters to 0 and print an alert.
    else:
        print('Unrecognized Character: ' + character)
        return(0)

# Create function to map numeric IDs to characters (reverse of 'char_to_num').
# 1 maps to 'a', 26 maps to 'z'.
# Any ID > 26 maps to other characters I'm not interested in.
# Any ID <= 0 maps to space.
def num_to_char(id):

    # Map non-zero 'id' to relevant character.
    if id > 0:
        return(chr(id + a_int - 1))

    # Map 'id' <= 0 to space.
    else:
        return(' ')

#######################
# CREATE TRAINING BATCH
#######################
# This section defines a class and functions to create a training batch
# for the RNN model








