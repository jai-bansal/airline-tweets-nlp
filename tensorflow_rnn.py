# This script builds a recurrent neural network (RNN) model
# for airline tweet data using the 'tensorflow' package.
# Specifically, it builds a long short term memory (LSTM) model.

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import pandas as pd
import string
import tensorflow as tf
import numpy as np
import random

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

#########################
# DEFINE MODEL PARAMETERS
#########################
# This section defines model parameters.

# Choose size of validation set.
valid_size = 1000

# Add variable for the length of the vocabulary size.
# This is 'a-z' plus space.
# This helps with saving some space below.
vocab_size = len(string.ascii_lowercase) + 1

# Set training batch size.
batch_size = 64

# Set length of each string in the batch (in characters).
batch_length = 10

# Set the number of nodes to be used in the LSTM matrix multiplications.
nodes = 64

#######################
# CREATE VALIDATION SET
#######################

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
# This section defines functions used to create a training batch
# for the LSTM model.
# I refer to a batch as some number ('batch_size') of strings of length ('batch_length').
# As opposed to referring to each character in the strings as a batch.
# It defines a new Python class to do this.
# This section is very specific to the Udacity example I learned from.

# Define new class that contains functions to generate training batch
# for LSTM model.
class BatchGenerator(object):

    # Define initialization values.
    def __init__(self, text, n_batch, len_batch):
        self._text = text                           # text
        self._n_batch = n_batch                     # batch size
        self._len_batch = len_batch                 # length of each string in batch

        # Define maximum length of each string in batch without overlap.
        max_string_length = len(text) // n_batch

        # Based on 'max_string_length', define starting positions for
        # each string in the batch. The starting positions occur every
        # 'max_string_length' characters apart.
        self._index = [shift * max_string_length for shift in range(n_batch)]

        # Initialize batches with strings of length 1.
        # '_next_character' function is defined below.
        self._initial_batch = self._next_character()

    # Define function that "gets" the next character for each string in the batch.
    def _next_character(self):

        # Create array of zeros with 1 row for each string in the batch and
        # 1 column for each character (a-z and space).
        next_char = np.zeros(shape = (self._n_batch, vocab_size),
                             dtype = np.float)

        # Loop through each string in the batch.
        # Set column corresponding to the "index" character to 1.
        for row in range(self._n_batch):

            # Set appropriate row and column (corresponding to current character) to 1.
            next_char[row, char_to_num(text[self._index[row]])] = 1.0

            # Increment relevant value of 'self._index'.
            self._index[row] = (self._index[row] + 1) % len(text)

        # Return 'next_char'.
        return(next_char)

    # Define function that runs '_next_character' 'len_batch' times to
    # generate a full batch of strings.
    def full_batch(self):

        # Start with initial character for each string in batch.
        batch = [self._initial_batch]

        # For 'len_batch' steps, add characters to 'batch' using the '_next_character' function.
        for character in range(len_batch):

            # Add characters to 'batch'.
            batch.append(self._next_character())

        # When done generating batch, reset 'self._initial_batch' to the last characters of 'batch'.
        self._initial_batch = batch[-1]
          
        # Return complete batch of strings.
        return(batch)

###############################
# CREATE OTHER HELPER FUNCTIONS
###############################
# This section defines some other functions that help with the LSTM model.

# The model outputs an array or bunch of arrays.
# This function turns those outputs into text.
# Define function to turn a 1-hot encoding or probability distribution
# into the (most likely) character.
def prob_dist_to_char(prob_dist):

    # Given a probability distribution over characters,
    # return the character corresponding to the column with the
    # highest probability.
    return[num_to_char(a) for a in np.argmax(prob_dist, 1)]

# Define function to compute log probability of the true labels in a batch.
# This value will help compute perplexity, a measure of how well the model performs.
# 'batch' is a (n x v) matrix where each row is a probability distribution
# on the next character in that string.
# 'labels' is also (n x v), but is a 1-hot encoding of the actual next character
# in each string (the answers).
# Then, low probabilities for the actual next character get a higher error
# than high probabilities for the actual next character.
def logprob(batch, labels):

    # Remove zero probabilities. I assume this is to avoid computing log(0).
    batch[batch < 1e-10] = 1e-10

    # Return the average negative log probabilities for the true
    # next character for the entire batch.
    return(np.sum(np.multiply(labels, -np.log(batch))) / labels.shape[0])

# Define function to sample 1 element from an array of normalized probabilities.
# Intuition: suppose 'prob_dist' is [0.2, 0.3, 0.5].
# 'r' will be < 0.2 20% of the time. Then, after the 1st iteration
# of the 'for' loop, 'prob_dist_crawler' >= r and 0 is returned.
# So, 0 is returned 20% of the time as desired.

# 'r' will be between 0.2 and 0.5 30% of the time. Then,
# 'prob_dist_crawler' >= r after the 2nd iteration of the 'for' loop.
# So 1 is returned 30% of the time as desired. Etc, etc.
def sample_prob_dist(prob_dist):

    # Randomly sample a value from uniform distribution (0, 1).
    r = random.uniform(0, 1)

    # Create variable that will "move along" 'prob_dist'.
    prob_dist_crawler = 0

    # Iteratively add the elements of 'prob_dist' to 'prob_dist_crawler'.
    # If at any point, 'prob_dist_crawler' >= r, return the corresponding
    # probability distribution element index. Otherwise, return
    # the index of the last value of the probability distribution.
    for i in range(len(prob_dist)):

        # Add a value of 'prob_dist' to 'prob_dist_crawler'.
        prob_dist_crawler += prob_dist[i]

        # If 'prob_dist_crawler' >= r, return corresponding
        # 'prob_dist' element index.
        if prob_dist_crawler >= r:
            return(i)

    # After looping through all values of 'prob_dist', if
    # 'prob_dist_crawler' was never >= r, return
    # index of the last value of the probability distribution.
    return(len(prob_dist) - 1)

# Define function to sample from a probability distribution (over the next character),
# take the index of the sampled value, and return an array that has
# all zeros except for the column corresponding to the index of the
# sampled value. This function uses 'sample_prob_dist' above.
def prob_dist_to_one_hot(prob_dist):

    # Create output array, initially with all zeros.
    one_hot = np.zeros(shape = [1, vocab_size],
                       dtype = np.float)

    # Sample value from 'prob_dist'. Turn the column in 'one_hot' that
    # corresponds to the index of the sampled value to 1.
    # 'prob_dist' and 'one_hot' should have the same number of columns.
    one_hot[0, sample_prob_dist(prob_dist[0])] = 1.0

    # Return 'one_hot'.
    return(one_hot)

# Define function to generate a random probability distribution.
# This will be used to generate the 1st letter of sentences below.
# After the 1st letter is randomly generated, the rest of the sentence
# is generated by the LSTM. This lets me see how the RNN is doing in terms of
# generating "reasonable" sentences.
def random_distribution():

    # Generate an array where each value is drawn from a uniform distribution (0, 1).
    random_dist = np.random.uniform(0.0, 1.0,
                                    size = [1, vocab_size])

    # Return a normalized version of 'random_dist' so it's a probability distribution.
    return(random_dist / np.sum(random_dist, 1))

###########
# RUN MODEL
###########
# This section runs the LSTM.

# Set up graph.
graph = tf.Graph()
with graph.as_default():

    # Create parameters. There are 3 gates: forget, input, and output
    # The input gate has a candidate function and input gate portion.
    # Each of the 4 things above has parameters for the current input,
    # previous output, and a bias term.
    # These are all trainable variables!
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is a great resource.

    fi = tf.Variable(tf.truncated_normal([vocab_size, nodes], -0.1, 0.1))   # forget gate current input
    fp = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))        # forget gate previous output
    fb = tf.Variable(tf.zeros([1, nodes]))                                  # forget gate bias
    
    ii = tf.Variable(tf.truncated_normal([vocab_size, nodes], -0.1, 0.1))   # input gate current input
    ip = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))        # input gate previous output
    ib = tf.Variable(tf.zeros([1, nodes]))                                  # input gate bias

    ci = tf.Variable(tf.truncated_normal([vocab_size, nodes], -0.1, 0.1))   # candidate function current input
    cp = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))        # candidate function previous output
    cb = tf.Variable(tf.zeros([1, nodes]))                                  # candidate function bias

    oi = tf.Variable(tf.truncated_normal([vocab_size, nodes], -0.1, 0.1))   # output gate current input
    op = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))        # output gate previous output
    ob = tf.Variable(tf.zeros([1, nodes]))                                  # output gate bias

    # Create non-trainable variables to hold the output and state over time.
    # These are initially set to all zeros.
    saved_output = tf.Variable(tf.zeros([batch_size, nodes]),
                               trainable = False)
    saved_state = tf.Variable(tf.zeros([batch_size, nodes]),
                              trainable = False)

    # The LSTM should return a probability distribution of length 27
    # which is 1 probability for each character.
    # The setup below returns a probability distribution of length
    # 'nodes'. So, the variables below are necessary for the transform.
    # Why not set 'nodes' so the output has the correct dimensions?
    # Not sure, but this is how the example I ripped from did it.
    transform_weights = tf.Variable(tf.truncated_normal([nodes, vocab_size], -0.1, 0.1))
    transform_biases = tf.Variable(tf.zeros([vocab_size]))
    

    
    





















