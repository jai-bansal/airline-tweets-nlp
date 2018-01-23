# This script builds a topic model using latent Dirichlet allocation (LDA).

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import pandas as pd
import nltk
from sklearn.decomposition import LatentDirichletAllocation as LDA

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
data.text = data.text.str.lower()

# Remove unwanted characters from 'text'.
# Note that I couldn't remove forward slashes.
data.text.replace(regex = True, inplace = True, to_replace = '@', value = '')  # '@' signs
data.text.replace(regex = True, inplace = True, to_replace = '\$', value = '')  # '$' signs
data.text.replace(regex = True, inplace = True, to_replace = ',', value = '')  # commas
data.text.replace(regex = True, inplace = True, to_replace = ':', value = '')  # colons
data.text.replace(regex = True, inplace = True, to_replace = ';', value = '')  # semi-colons
data.text.replace(regex = True, inplace = True, to_replace = '-', value = '')  # minus signs
data.text.replace(regex = True, inplace = True, to_replace = '\+', value = '')  # plus signs
data.text.replace(regex = True, inplace = True, to_replace = '#', value = '')  # hashtags
data.text.replace(regex = True, inplace = True, to_replace = '\?', value = '')  # question marks
data.text.replace(regex = True, inplace = True, to_replace = '\!', value = '')  # question marks
data.text.replace(regex = True, inplace = True, to_replace = '\.', value = '')  # periods
data.text.replace(regex = True, inplace = True, to_replace = '\'', value = '')  # apostrophes
data.text.replace(regex = True, inplace = True, to_replace = '\"', value = '')  # quotes
data.text.replace(regex = True, inplace = True, to_replace = '\/', value = '')  # backslashes
data.text.replace(regex = True, inplace = True, to_replace = '\&', value = '')  # '&' signs
data.text.replace(regex = True, inplace = True, to_replace = '\(', value = '')  # parentheses
data.text.replace(regex = True, inplace = True, to_replace = '\)', value = '')  # parentheses
data.text.replace(regex = True, inplace = True, to_replace = '\d+', value = '')  # all digits
data.text.replace(regex = True, inplace = True, to_replace = 'virginamerica', value = '')  # airline tag
data.text.replace(regex = True, inplace = True, to_replace = 'americanair', value = '')  # airline tag
data.text.replace(regex = True, inplace = True, to_replace = 'united', value = '')  # airline tag
data.text.replace(regex = True, inplace = True, to_replace = 'southwestair', value = '')  # airline tag
data.text.replace(regex = True, inplace = True, to_replace = 'jetblue', value = '')  # airline tag
data.text.replace(regex = True, inplace = True, to_replace = 'usairways', value = '')  # airline tag
data.text = data.text.str.strip() # remove leading and trailing spaces
data.text.replace(regex = True, inplace = True, to_replace = '  ', value = ' ')  # double spaces with spaces

# Download stopwords and tokenizer help (only needs to be done once).
#nltk.download('stopwords')
#nltk.download('punkt')

# Tokenize words in 'data.text' (put words into list).
data['tokenized_text'] = data.text.apply(nltk.word_tokenize)

# Specify (English) stopwords.
stop_words = set(nltk.corpus.stopwords.words('english'))

# Create column in 'data' for cleaned, tokenized strings.
data['clean_tokens'] = data.tokenized_text.apply(lambda x: [word for word in x if word not in stop_words])

#############
# CONDUCT LDA
#############

# Create LDA object with 5 topics.
lda = LDA(n_components = 5,
          random_state = 20180122)

