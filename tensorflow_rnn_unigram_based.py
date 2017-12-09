# This script builds a recurrent neural network (RNN) model
# for airline tweet data using the 'tensorflow' package.
# Specifically, it builds a long short term memory (LSTM) model.
# The goal of the model is to predict the next character in a string.
# The model in this script is trained over unigrams; it trains on
# one previous character to predict the next.

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

# Set size (number of columns) of embeddings.
embedding_size = 128

# Set number of steps to run LSTM model.
steps = 10000

# Set step interval for printing loss.
loss_interval = 500

# Set step interval for printing example sentences.
example_interval = 500

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
        # print('Unrecognized Character: ' + character) # commented out to suppress output
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
# This section defines functions used to create a training batch for the LSTM model.
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
            next_char[row, char_to_num(self._text[self._index[row]])] = 1.0

            # Increment relevant value of 'self._index'.
            self._index[row] = (self._index[row] + 1) % len(self._text)

        # Return 'next_char'.
        return(next_char)

    # Define function that runs '_next_character' 'len_batch' times to
    # generate a full batch of strings.
    def full_batch(self):

        # Start with initial character for each string in batch.
        batch = [self._initial_batch]

        # For 'len_batch' steps, add characters to 'batch' using the '_next_character' function.
        for character in range(self._len_batch):

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
# So, it returns a one-hot vector for the sampled character.
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
# This generates the 1st letter of example sentences below.
# After the 1st letter is randomly generated, the rest of the sentence
# is generated by the LSTM. This shows how the RNN is doing in terms of
# generating "reasonable" sentences.
def random_distribution():

    # Generate an array where each value is drawn from a uniform distribution (0, 1).
    random_dist = np.random.uniform(0.0, 1.0,
                                    size = [1, vocab_size])

    # Return a normalized version of 'random_dist' so it's a probability distribution.
    return(random_dist / np.sum(random_dist, 1))

##############
# SET UP GRAPH
##############
# This section sets up the graph for the LSTM.
# The goal of the model is to predict the next character in a string.

# Set up graph.
graph = tf.Graph()
with graph.as_default():

    # Create embeddings (1 embedding for every character in the vocabulary).
    embeddings = tf.Variable(tf.random_uniform(shape = [vocab_size, embedding_size],
                                               minval = -1,
                                               maxval = 1))

    # Create parameters. There are 3 gates: forget, input, and output
    # The input gate has a candidate function and input gate portion.
    # Each of the 4 things above has parameters for the current input,
    # previous output, and a bias term.
    # These are all trainable variables!
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is a great resource.

    fi = tf.Variable(tf.truncated_normal([embedding_size, nodes], -0.1, 0.1))   # forget gate current input
    fp = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))            # forget gate previous output
    fb = tf.Variable(tf.zeros([1, nodes]))                                      # forget gate bias

    ci = tf.Variable(tf.truncated_normal([embedding_size, nodes], -0.1, 0.1))   # candidate function current input
    cp = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))            # candidate function previous output
    cb = tf.Variable(tf.zeros([1, nodes]))                                      # candidate function bias

    ii = tf.Variable(tf.truncated_normal([embedding_size, nodes], -0.1, 0.1))   # input gate current input
    ip = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))            # input gate previous output
    ib = tf.Variable(tf.zeros([1, nodes]))                                      # input gate bias

    oi = tf.Variable(tf.truncated_normal([embedding_size, nodes], -0.1, 0.1))   # output gate current input
    op = tf.Variable(tf.truncated_normal([nodes, nodes], -0.1, 0.1))            # output gate previous output
    ob = tf.Variable(tf.zeros([1, nodes]))                                      # output gate bias

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
    # Not sure, but this is how the Udacity example I ripped from did it.
    transform_weights = tf.Variable(tf.truncated_normal([nodes, vocab_size], -0.1, 0.1))
    transform_biases = tf.Variable(tf.zeros([vocab_size]))

    # Define function to execute LSTM matrix multiplications and other operations.
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/ is a great resource.
    def lstm_cell(current_input, previous_output, cell_state):

        # Create forget gate.
        forget_gate = tf.sigmoid(tf.matmul(current_input, fi) + tf.matmul(previous_output, fp) + fb)

        # Create candidate input function.
        candidate = tf.matmul(current_input, ci) + tf.matmul(previous_output, cp) + cb

        # Create input gate.
        input_gate = tf.sigmoid(tf.matmul(current_input, ii) + tf.matmul(previous_output, ip) + ib)

        # Create output gate.
        output_gate = tf.sigmoid(tf.matmul(current_input, oi) + tf.matmul(previous_output, op) + ob)

        # Update cell state.
        cell_state = (forget_gate * cell_state) + (input_gate * tf.tanh(candidate))

        # Return the output and updated cell state.
        return((output_gate * tf.tanh(cell_state)), cell_state)

    # Create structure for training data.

    # Create empty list for training data.
    train_data = list()

    # Fill 'train_data' with placeholders of appropriate dimensions.
    # These will be filled in when the model runs with actual training data.
    for _ in range(batch_length + 1):

        # Add placeholders to 'train_data'.
        train_data.append(tf.placeholder(tf.float32,
                                         shape = [batch_size, vocab_size]))

    # Define training data inputs (all characters in 'train_data' except the last one).
    train_inputs = train_data[:batch_length]

    # Define training data labels (all characters in 'train_data' except the first one).
    train_labels = train_data[1:]

    # Create empty list to hold LSTM outputs (predictions).
    outputs = list()
        
    # Create variables to hold the previous output and current cell state.
    # These seem redundant because of 'saved_output' and 'saved_state' above.
    # But those 2 are only updated at the end of each batch.
    # The 2 variables below are updated "iteration in, iteration out".
    previous_output = saved_output
    cell_state = saved_state

    # Get the corresponding embedding for each training example (character).
    # Plug embedding into LSTM (computations also defined above).
    # Update the output and cell state. Save results.
    for i in train_inputs:

        # Get corresponding embedding for 'i'.
        i_embedding = tf.nn.embedding_lookup(embeddings,
                                             tf.argmax(i, dimension = 1))

        # Run LSTM cell and update 'previous_output' and 'cell_state'.
        previous_output, cell_state = lstm_cell(i_embedding, previous_output, cell_state)

        # Add LSTM output to 'outputs'.
        outputs.append(previous_output)

    # Compute loss. Before this, a few things must happen.

    # Update 'saved_output' and 'saved_state' to be ready for the next batch.
    # The setup below means that the updating will occur before the 'logit' and 'loss' steps.
    with tf.control_dependencies([saved_output.assign(previous_output),
                                  saved_state.assign(cell_state)]):

        # Recall from above that the model output has incorrect dimensions.
        # Transform model output into correct dimensions.
        # The 'concat' combines the 'batch_length' sets of 'batch_size' predictions and is also done
        # for the training labels below. So 'final_outputs' has dimensions
        #('batch_size' x 'batch_length') x 'vocab_size'.
        final_outputs = tf.nn.xw_plus_b(x = tf.concat(0, outputs),
                                        weights = transform_weights,
                                        biases = transform_biases)

        # Compute loss. Note that the 'batch_length' sets of training labels are combined.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat(0, train_labels),
                                                                      logits = final_outputs))

    # Specify optimizer with decaying learning rate.

    # Create a global step variable to track steps to help with the decaying learning rate.
    global_step = tf.Variable(0)

    # Specify exponentially decaying learning rate.
    learning_rate = tf.train.exponential_decay(10.0,
                                               global_step,
                                               5000,
                                               0.1,
                                               staircase = True)

    # Specify optimizer using 'learning_rate'.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # Work with the gradients.
    # Computation and application of gradients is done separately because
    # the gradients are clipped between computation and application.
    # Not sure of all the details in this section, but this works!

    # Compute gradients.
    gradients, variables = zip(*optimizer.compute_gradients(loss))

    # Clip gradients.
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)

    # Apply clipped gradients to variables.
    optimizer = optimizer.apply_gradients(zip(gradients, variables),
                                          global_step = global_step)

    # Generate predictions.
    train_preds = tf.nn.softmax(final_outputs)

    # Handle the generation of example text.
    # The code below helps generate random sentences.
    # This helps subjectively judge how well the model is doing.

    # Define example input (1 character).
    example_input = tf.placeholder(tf.float32,
                                   shape = [1, vocab_size])

    # Look up corresponding embedding for 'example_input'.
    example_input_embedding = tf.nn.embedding_lookup(embeddings,
                                                     tf.argmax(example_input, dimension = 1))

    # For example text, set initial state and previous output to all zeros.
    # In the Udacity example, there is NO 'trainable = False'.
    # But I think it should be in there...these shouldn't be trained!
    # 'saved_output' and 'saved_state' above have 'trainable = False'.
    saved_example_output = tf.Variable(tf.zeros([1, nodes]),
                                       trainable = False)
    saved_example_cell_state = tf.Variable(tf.zeros([1, nodes]),
                                           trainable = False)

    # Create an operation to reset 'saved_example_output' and 'saved_example_state'
    # to zeros.
    # This is used below after each set of 5 example sentences is printed.
    reset_example_output_and_state = tf.group(saved_example_output.assign(tf.zeros([1, nodes])),
                                              saved_example_cell_state.assign(tf.zeros([1, nodes])))

    # Run a letter of the example text through the LSTM cell.
    # I later generate a prediction for the next letter using the output of the LSTM cell.
    example_output, example_cell_state = lstm_cell(example_input_embedding, saved_example_output, saved_example_cell_state)

    # Generate prediction for the next character.
    # But first update 'saved_example_output' and 'saved_example_state' for the next iteration.
    with tf.control_dependencies([saved_example_output.assign(example_output),
                                  saved_example_cell_state.assign(example_cell_state)]):
        example_pred = tf.nn.softmax(tf.nn.xw_plus_b(x = example_output,
                                                     weights = transform_weights,
                                                     biases = transform_biases))

###########
# RUN GRAPH
###########
# This section runs the graph for the LSTM model.

# Create initial batch generation object.
train_batches = BatchGenerator(train_text, 64, 10)

# Create validation set batch object.
valid_batches = BatchGenerator(valid_text, 1, 1)
  
# Run graph.
with tf.Session(graph = graph) as session:

    # Initialize variables.
    tf.global_variables_initializer().run()

    # Initialize loss tracker at 0.
    loss_tracker = 0

    # Iterate through steps.
    for step in range(steps):

        # Generate training batch.
        batch = train_batches.full_batch()

        # Create empty dictionary that will be populated and fed into LSTM model.
        feed_dict = dict()

        # Populate 'feed_dict' with the elements of 'batch'.
        for i in range(batch_length + 1):

            feed_dict[train_data[i]] = batch[i]

        # Run elements of 'batch' through the LSTM.
        _, l, preds, lr = session.run([optimizer, loss, train_preds, learning_rate],
                                      feed_dict = feed_dict)

        # Add 'l' (loss from above) to 'loss_tracker'.
        loss_tracker += l

        # Every 'loss_interval' steps, report some metrics.
        if step > 0 and step % loss_interval == 0:

            # Print the average loss over the last 'loss_interval' steps.
            print('Step', step, ':')
            print('Mean Loss =', round(loss_tracker / loss_interval, 2))
            print('Learning Rate =', lr)

            # Reset 'loss_tracker' to 0.
            loss_tracker = 0

            # Print batch perplexity.
            # Perplexity is a measure of how well the model predicts
            # a sample (lower is better).

            # Get the labels from 'batch' (all elements except the 1st one).
            batch_labels = np.concatenate(list(batch)[1:])

            # Print batch perplexity ('logprob' is a function defined above).
            print('Batch Perplexity =',
                  round(np.exp(logprob(preds, batch_labels)), 2))

            # Print validation set perplexity.

            # 'reset_sample_output_and_state' is defined above.
            # Before computing validation set perplexity, set 'saved_sample_output'
            # and 'saved_sample_cell_state' to zeros.
            reset_example_output_and_state.run()

            # Set initial validation set log probability counter to 0.
            # The log probability from each validation set example will be
            # added and eventually converted to perplexity.
            valid_logprob_counter = 0

            # Iterate through the validation set('valid_text').
            for _ in range(valid_size):

                # Generate validation set batch.
                valid_batch = valid_batches.full_batch()

                # Generate prediction for next character using 'valid_batch[0]'
                # as input.
                valid_pred = example_pred.eval({example_input: valid_batch[0]})

                # Compute the log probability between 'valid_pred' and the actual
                # next character in the validation set (valid_batch[1]).
                # Add this to 'valid_logprob_counter'.
                valid_logprob_counter = valid_logprob_counter + logprob(valid_pred, valid_batch[1])

            # Print validation set perplexity.
            print('Validation Set Perplexity = ',
                  round(np.exp(valid_logprob_counter / valid_size), 2))
            print('')

            # Every 'example_interval' steps, generate 5 sample sentences.
            if step % example_interval == 0:
            
                # One iteration for each sentence.
                for _ in range(5):

                    # 'reset_sample_output_and_state' is defined above.
                    # At the beginning of each sentence, set 'saved_sample_output'
                    # and 'saved_sample_cell_state' to zeros.
                    reset_example_output_and_state.run()

                    # Randomly generate 1st letter of sentence.
                    # 'random_distribution' and 'prob_dist_to_one_hot' are
                    # functions defined above. The 1st function creates a
                    # random distribution and the 2nd function samples from it.
                    # 'letter' is a one-hot vector.
                    # This uses the same general mechanism used for computing
                    # validation set perplexity.
                    letter_one_hot = prob_dist_to_one_hot(random_distribution())

                    # Start the sentence. 'letter_one_hot' is a one-hot vector
                    # and must be turned into an actual letter.
                    example_sentence = prob_dist_to_char(letter_one_hot)[0]

                    # Generate 79 more characters for the sentence.
                    for _ in range(79):

                        # Generate 'example_pred' by using 'letter_one_hot' as
                        # the input to the LSTM. Save output and cell state and
                        # keep iterating to generate new letters for the example sentence.
                        next_letter_distribution = example_pred.eval({example_input: letter_one_hot})

                        # Update 'letter_one_hot'.
                        # 'next_letter_distribution' is a probability distribution over characters.
                        letter_one_hot = prob_dist_to_one_hot(next_letter_distribution)

                        # Add 'letter_one_hot' in letter form to 'example_sentence'.
                        example_sentence += prob_dist_to_char(letter_one_hot)[0]

                    # Print 'example_sentence'.
                    print(example_sentence)

                print('')




            
        

    
    





















