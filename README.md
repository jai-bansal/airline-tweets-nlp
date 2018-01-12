#### Synopsis:
This project contains examples of natural language processing (NLP) analysis on Twitter airline tweets from part of February 2015 in R and Python. Work includes:

- a word cloud
- sentiment analysis models
- skip-gram and continuous bag-of-word (CBOW) models
- a long short term memory recurrent neural network (LSTM)
- topic modeling using latent Dirichlet allocation (LDA)

#### Motivation:
To learn about NLP in R and Python.

#### Contents:
The R and Python files are located in the 'R' and 'Python' branches respectively. <br />
The dataset ('Airline-Sentiment-2-w-AA.csv') is included in both branches.

Some analyses are only included in R or Python. For example, only the Python branch has skip-gram and CBOW scripts.

The files in each branch are mostly intuitively named, except for...

The R branch includes the RMarkdown script containing the word cloud and sentiment analysis models ('airline_tweet_analysis.Rmd'). <br />
It also contains a PDF file ('airline_tweet_analysis.pdf') that is the product (with some intermediate editing) of running the RMarkdown script.

The Python branch contains a Jupyter Notebook file ('airline_tweet_analysis.ipynb') with the word cloud and sentiment analysis models.
There is also an image file ('airline_tweet_analysis_wordcloud.png') of a wordcloud of the cleaned data since this image does not show up in the Jupyter notebook.

#### Dataset Details:
I use the 'Airline Twitter sentiment' dataset obtained from: https://www.crowdflower.com/data-for-everyone/. To find the dataset, search for 'Airline' on the page.

Specifically, I use the 16,000 row dataset uploaded on 2015-02-12 by CrowdFlower. <br />
I assume the upload date is incorrect as the data includes tweets from after 2015-02-12... <br />
Note that the actual dataset only appears to contain 14,640 rows...not sure where the discrepancy between actual rows and rows stated on the website comes from, but it doesn't affect my analysis. <br />
This dataset is a scrape of Twitter data from part of February 2015. <br />
The included tweets are about major US airlines. <br />
Contributors were asked to classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service"). <br />
The tweet classification provides 'answers' for sentiment analysis, allowing me to conduct more interesting analysis and prediction.

#### License:
GNU General Public License
