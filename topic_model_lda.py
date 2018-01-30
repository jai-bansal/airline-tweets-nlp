# This script builds a topic model using latent Dirichlet allocation (LDA).
# I use the 'sklearn' module.
# It's not as easy to use as the R version.
# There's also another module (not included) called 'gensim'.
# It was too involved to get something running in 'gensim' for the purposes
# of this fast and easy script.

################
# IMPORT MODULES
################
# This section loads modules.

# Import modules.
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
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

#####
# LDA
#####

# Create TF-IDF matrix object.
tfidf = CountVectorizer(analyzer = 'word',
                        strip_accents = 'unicode',
                        stop_words = 'english',
                        lowercase = True,
                        max_df = 0.95,
                        min_df = 5)

# Convert 'data.text' to TF-IDF matrix.
data_tfidf = tfidf.fit_transform(data.text)

# Create LDA object with 5 topics.
lda = LDA(n_components = 5,
          random_state = 20180122,
          learning_method = 'batch',
          verbose = 5)

# Fit LDA model.
lda.fit(data_tfidf)

# View topic pseudo-counts for each word.
lda.components_
