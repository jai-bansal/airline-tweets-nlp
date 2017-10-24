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
import random as rand
import math

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
# A skip-gram model picks a word in the center of a string to
# start and uses another word in the string as the target.

# Create index variable to track position in 'string_indices'.
string_index = 0

# Define function to generate training batch for skip-gram model.
# 'batch_size' is the size of the training batch.
# 'start_word_index' is the index of the starting word.
# It is also half of the words available for sampling as targets.
# 'samples' is the number of targets picked for each 'start_word_index'.
# 'string_index' tracks the relevant position in 'string_indices'.
# Note that 'batch_size' / 'samples' should be a positive integer.
# Note that 'samples' <= 2 * 'start_word_index' must hold.
# Otherwise, I'm asking the model to sample more words than are available.
def generate_training_batch(batch_size, start_word_index, samples, string_index):

    # Create 'batch' and 'labels'...initial values don't matter as they are later replaced.
    inputs = np.ndarray(shape = batch_size,
                        dtype = np.int32)
    labels = np.ndarray(shape = (batch_size, 1),
                        dtype = np.int32)

    # Define relevant substring length.
    # The relevant substring is the starting word and the words to either side
    # available for sampling as targets.
    substring_length = (2 * start_word_index) + 1

    # Set up a list-like object with maximum length of 'substring_length'.
    # This will hold all words of the relevant substring.
    substring = collections.deque(maxlen = substring_length)

    # Fill 'substring' with elements of 'string_indices' starting at
    # 'string_index' and adding 'substring_length' elements.
    for _ in range(substring_length):

        # Add element of 'string_indices' to 'substring'.
        substring.append(string_indices[string_index])

        # Increment 'string_index'.
        string_index = (string_index + 1) % len(string_indices)

    # Loop through starting words of model.
    for i in range(batch_size / samples):

        # Initialize 'target' variable.
        # This will be the target for the model.
        # I set it as 'start_word_index' now, but it will change.
        # 'start_word_index' is the training example, so it can't be the target!
        target = start_word_index

        # Skip-gram starts with the word in the center of 'substring'
        # and uses another word in 'substring' as target.
        # So the word in the center of 'substring' can't be the target.
        # Create a list for indices to avoid as the target.
        to_avoid = [start_word_index]

        # For each starting word, loop through the 'samples'.
        for j in range(samples):

            # If 'target' is in 'to_avoid', resample 'target'.
            while target in to_avoid:
                target = rand.randint(0, (substring_length - 1))

            # Add 'target' to 'to_avoid' so this particular training example isn't repeated.
            to_avoid.append(target)

            # Fill in 'train_examples' and 'labels'.
            inputs[(i * samples)+ j] = substring[start_word_index]
            labels[(i * samples) + j] = substring[target]

        # Since we set a maximum length for 'substring',
        # the 1st element is dropped and 'string_indices[string_index]' is added to the end.
        # This is still in the 'i' loop!
        # Conveyor belt idea.
        substring.append(string_indices[string_index])
        string_index = (string_index + 1) % len(string_indices)

    # Return function outputs.
    return(inputs, labels, string_index)

######################
# SET MODEL PARAMETERS
######################
# This section sets parameters for the skip-gram model.

# Set batch size.
batch_size = 128

# Set embedding size (# of features per vocabulary word).
embedding_size = 128

# Set how many words to consider left and right.
# So this is half of the total words to consider for sampling as targets.
# This also turns out to be the index of the starting word.
start_word_index = 1

# Set the number of samples to use for one starting word.
# Another way to say it: select the number of labels to generate for one input.
samples = 2

# Set size of validation set.
valid_size = 16

# Pick validation set indices.
# The corresponding words will have their nearest neighbors sampled.
# I only consider the 100 most common words for the validation set.
# These are the first 100 entries in 'word_dict'.
# 'word_dict' was created such that sampling between 0 and 99 gets the
# 100 most common words I want.
validation_set = rand.sample(range(100),
                             valid_size)

# There are many possible output classes (5000).
# Training the full model would require computing the softmax
# for each class for each training example, which is slow.
# So, for each training batch, I only consider a randomly chosen
# subset of classes.
# The number of classes randomly sampled is defined below.
classes_sampled = 64

###########
# RUN MODEL
###########
# This section runs the skip-gram model.

# Set up graph.
graph = tf.Graph()
with graph.as_default():

    # Input data.
    inputs_holder = tf.placeholder(tf.int32,
                                   shape = batch_size)
    labels_holder = tf.placeholder(tf.int32,
                                   shape = [batch_size, 1])
    validation_set_tf = tf.constant(validation_set,
                                    dtype = tf.int32)

    # Define embeddings variable.
    # These are the (initially random) features for each vocabulary word.
    embeddings = tf.Variable(tf.random_uniform(shape = [vocab_size, embedding_size],
                                               minval = -1.0,
                                               maxval = 1.0))

    # Define weights.
    # Dimensions are defined by 'tf.nn.sampled_softmax_loss' used below.
    weights = tf.Variable(tf.truncated_normal(shape = [vocab_size, embedding_size],
                                              stddev = 1.0 / math.sqrt(embedding_size)))

    # Define biases.
    # Dimensions are defined by 'tf.nn.sampled_softmax_loss' used below.
    biases = tf.Variable(tf.zeros([vocab_size]))

    # Get embeddings for inputs.
    input_embeddings = tf.nn.embedding_lookup(embeddings,
                                              inputs_holder)

    # Define loss function.
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = weights,
                                                     biases = biases,
                                                     inputs = input_embeddings,
                                                     labels = labels_holder,
                                                     num_sampled = classes_sampled,
                                                     num_classes = vocab_size))

    # Define optimizer.
    # 'embeddings', 'weights', and 'biases' are optimized since they're
    # all defined as variables above.
    # Why this particular optimizer? Good question, that's the one in the example I'm using.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    #






























