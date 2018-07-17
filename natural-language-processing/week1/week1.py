# coding: utf-8

# # Predict tags on StackOverflow with linear models

# In this assignment you will learn how to predict tags for posts from [StackOverflow](https://stackoverflow.com). To solve this task you will use multilabel classification approach.
# 
# ### Libraries
# 
# In this task you will need the following libraries:
# - [Numpy](http://www.numpy.org) — a package for scientific computing.
# - [Pandas](https://pandas.pydata.org) — a library providing high-performance, easy-to-use data structures and data analysis tools for the Python
# - [scikit-learn](http://scikit-learn.org/stable/index.html) — a tool for data mining and data analysis.
# - [NLTK](http://www.nltk.org) — a platform to work with natural language.

# ### Data
# 
# The following cell will download all data required for this assignment into the folder `week1/data`.

# In[ ]:


import sys
import os

sys.path.append("..")

# from common.download_utils import download_week1_resources

#download_week1_resources()

# ### Grading
# We will create a grader instance below and use it to collect your answers. Note that these outputs will be stored locally inside grader and will be uploaded to platform only after running submitting function in the last part of this assignment. If you want to make partial submission, you can run that cell any time you want.

# In[1]:


from grader import Grader
grader = Grader()


import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords


from ast import literal_eval
import pandas as pd
import numpy as np





def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data



train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')



# train.head()


X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values


import re
from nltk.tokenize import word_tokenize


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def text_prepare(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text

    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    tokens = word_tokenize(text)
    text = [t for t in tokens if t not in STOPWORDS]  # delete stopwords from text
    return " ".join(text).strip()





def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function",
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# print(test_text_prepare())


prepared_questions = []
for line in open('data/text_prepare_tests.tsv'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

grader.submit_tag('TextPrepare', text_prepare_results)


X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]



print X_train[:3]
print y_train




# Dictionary of all tags from train corpus with their counts.
tags_counts = {}
# Dictionary of all words from train corpus with their counts.
words_counts = {}


######################################
######### YOUR CODE HERE #############
######################################

def countWords(txt):
    counts = {}
    for x in txt:
        tokens = word_tokenize(x)
        for token in tokens:
            if not token in counts:
                counts[token] = 1
            else:
                counts[token] += 1
    return counts


words_counts = countWords(X_train)





def countTags(A):
    dic = {}
    for x in A:
        if not x in dic:
            dic[x] = A.count(x)
    return dic


flat_list = [item for sublist in y_train for item in sublist]
tags_counts = countTags(flat_list)

# In[18]:


most_common_tags = sorted(tags_counts.iteritems(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.iteritems(), key=lambda x: x[1], reverse=True)[:3]


# We are assuming that *tags_counts* and *words_counts* are dictionaries like `{'some_word_or_tag': frequency}`. After applying the sorting procedure, results will be look like this: `[('most_popular_word_or_tag', frequency), ('less_popular_word_or_tag', frequency), ...]`. The grader gets the results in the following format (two comma-separated strings with line break):
# 
#     tag1,tag2,tag3
#     word1,word2,word3
# 
# Pay attention that in this assignment you should not submit frequencies or some additional information.

# In[19]:


most_common_tags = sorted(tags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)[:3]

grader.submit_tag('WordsTagsCount', '%s\n%s' % (','.join(tag for tag, _ in most_common_tags),
                                                ','.join(word for word, _ in most_common_words)))


# ### Transforming text to a vector
# 
# Machine Learning algorithms work with numeric data and we cannot use the provided text data "as is". There are many ways to transform text data to numeric vectors. In this task you will try to use two of them.
# 
# #### Bag of words
# 
# One of the well-known approaches is a *bag-of-words* representation. To create this transformation, follow the steps:
# 1. Find *N* most popular words in train corpus and numerate them. Now we have a dictionary of the most popular words.
# 2. For each title in the corpora create a zero vector with the dimension equals to *N*.
# 3. For each text in the corpora iterate over words which are in the dictionary and increase by 1 the corresponding coordinate.
# 
# Let's try to do it for a toy example. Imagine that we have *N* = 4 and the list of the most popular words is 
# 
#     ['hi', 'you', 'me', 'are']
# 
# Then we need to numerate them, for example, like this: 
# 
#     {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
# 
# And we have the text, which we want to transform to the vector:
# 
#     'hi how are you'
# 
# For this text we create a corresponding zero vector 
# 
#     [0, 0, 0, 0]
#     
# And iterate over all words, and if the word is in the dictionary, we increase the value of the corresponding position in the vector:
# 
#     'hi':  [1, 0, 0, 0]
#     'how': [1, 0, 0, 0] # word 'how' is not in our dictionary
#     'are': [1, 0, 0, 1]
#     'you': [1, 1, 0, 1]
# 
# The resulting vector will be 
# 
#     [1, 1, 0, 1]
#    
# Implement the described encoding in the function *my_bag_of_words* with the size of the dictionary equals to 5000. To find the most common words use train data. You can test your code using the function *test_my_bag_of_words*.

# In[20]:


def get_w2i(word_counts):
    popular_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    keys = list(popular_words)
    i = 0
    dic = {}
    while i < len(keys):
        dic[keys[i]] = i
        i += 1
    return dic


# In[21]:


def get_i2w(w2i):
    return list(w2i.keys())


# In[22]:


DICT_SIZE = 5000
WORDS_TO_INDEX = get_w2i(words_counts)  ####### YOUR CODE HERE #######
INDEX_TO_WORDS = get_i2w(WORDS_TO_INDEX)  ####### YOUR CODE HERE #######
ALL_WORDS = WORDS_TO_INDEX.keys()


def my_bag_of_words(text, words_to_index, dict_size):
    """
        text: a string
        dict_size: size of the dictionary

        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = list(np.zeros(dict_size, dtype=np.int))


    if isinstance(text, str) == True:
        text = text.split()

    for txt in text:
        words = word_tokenize(txt)
        for w in words:
            if w in words_to_index:
                i = words_to_index[w]
                result_vector[i] = int(1 + result_vector[i])

    # print result_vector
    return result_vector


# In[23]:


#print my_bag_of_words(['hi how are you'], {'me': 2, 'you': 1, 'hi': 0, 'are': 3}, 4)


# In[24]:


def test_my_bag_of_words():
    words_to_index = {'hi': 0, 'you': 1, 'me': 2, 'are': 3}
    examples = ['hi how are you']
    answers = [[1, 1, 0, 1]]
    for ex, ans in zip(examples, answers):
        print ex, ans, words_to_index
        if (my_bag_of_words(ex, words_to_index, 4) != ans):
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'


# In[25]:


print(test_my_bag_of_words())

# Now apply the implemented function to all samples (this might take up to a minute):

# In[26]:


from scipy import sparse as sp_sparse
import pickle


def get_x_train_my_bag():

    try:
        x_train_mybag_fileObject = open("x_train_mybag", 'r')
        X_train_mybag = pickle.load(x_train_mybag_fileObject)
    except:
        X_train_mybag = None

    if X_train_mybag is None:
        X_train_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_train])

        fileObject = open("x_train_mybag", 'wb')
        pickle.dump(X_train_mybag, fileObject)

    return X_train_mybag


def get_X_val_mybag():

    try:
        X_val_mybag_fileObject = open("X_val_mybag", 'r')
        X_val_mybag = pickle.load(X_val_mybag_fileObject)
    except:
        X_val_mybag = None

    if X_val_mybag is None:
        X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])

        fileObject = open("X_val_mybag", 'wb')
        pickle.dump(X_val_mybag, fileObject)

    return X_val_mybag


def get_X_test_mybag():

    try:
        X_test_mybag_fileObject = open("X_test_mybag", 'r')
        X_test_mybag = pickle.load(X_test_mybag_fileObject)
    except:
        X_test_mybag = None

    if X_test_mybag is None:
        X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

        fileObject = open("X_test_mybag", 'wb')
        pickle.dump(X_test_mybag, fileObject)

    return X_test_mybag

X_train_mybag = get_x_train_my_bag()
X_val_mybag = get_X_val_mybag()
X_test_mybag = get_X_test_mybag()

print('X_train shape ', X_train_mybag.shape)
print('X_val shape ', X_val_mybag.shape)
print('X_test shape ', X_test_mybag.shape)


#X_val_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_val])


#X_test_mybag = sp_sparse.vstack([sp_sparse.csr_matrix(my_bag_of_words(text, WORDS_TO_INDEX, DICT_SIZE)) for text in X_test])

# As you might notice, we transform the data to sparse representation, to store the useful information efficiently. There are many [types](https://docs.scipy.org/doc/scipy/reference/sparse.html) of such representations, however sklearn algorithms can work only with [csr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix) matrix, so we will use this one.

# **Task 3 (BagOfWords).** For the 11th row in *X_train_mybag* find how many non-zero elements it has.
# In this task the answer (variable *non_zero_elements_count*) should be a number, e.g. 20.

# In[ ]:

counter = lambda lst: (sum(1 for el in lst if el != 0))

row = X_train_mybag[10].toarray()[0]

print row

print row.shape

non_zero_elements_count = counter(row) ####### YOUR CODE HERE #######
print non_zero_elements_count

grader.submit_tag('BagOfWords', str(non_zero_elements_count))

# #### TF-IDF
#
# The second approach extends the bag-of-words framework by taking into account total frequencies of words in the corpora.
# It helps to penalize too frequent words and provide better features space.
#
# Implement function *tfidf_features* using class
# [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
# from *scikit-learn*. Use *train* corpus to train a vectorizer. Don't forget to take a look into the arguments
# that you can pass to it. We suggest that you filter out too rare words (occur less than in 5 titles) and
# too frequent words (occur more than in 90% of the titles). Also, use bigrams along with unigrams in your vocabulary.




from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


def tfidf_features(X_train, X_val, X_test):
    """
        X_train, X_val, X_test — samples
        return TF-IDF vectorized representation of each sample and vocabulary
    """
    # Create TF-IDF vectorizer with a proper parameters choice
    # Fit the vectorizer on the train set
    # Transform the train, test, and val sets and return the result

    tfidf_vectorizer = TfidfVectorizer(token_pattern='(\S+)')

    ######################################
    ######### YOUR CODE HERE #############
    ######################################

    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)

    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_

#
# # Once you have done text preprocessing, always have a look at the results.
# Be very careful at this step, because the performance of future models will drastically depend on it.
# #
# # In this case, check whether you have c++ or c# in your vocabulary,
# as they are obviously important tokens in our tags prediction task:
#
# # In[ ]:
#
#
X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)
tfidf_reversed_vocab = {i: word for word, i in tfidf_vocab.items()}

print tfidf_reversed_vocab

#
# # In[ ]:
#
#
# ######### YOUR CODE HERE #############
#
#
# # If you can't find it, we need to understand how did it happen that we lost them? It happened during the built-in
# tokenization of TfidfVectorizer. Luckily, we can influence on this process.
# Get back to the function above and use '(\S+)' regexp as a *token_pattern* in the constructor of the vectorizer.
#
# # Now, use this transormation for the data and check again.
#
# # In[ ]:
#
#
# ######### YOUR CODE HERE #############
#
#
# # ### MultiLabel classifier
# #
# # As we have noticed before, in this task each example can have multiple tags.
# To deal with such kind of prediction, we need to transform labels in a binary form and
# the prediction will be a mask of 0s and 1s. For this purpose it is convenient
# to use [MultiLabelBinarizer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html)
# from *sklearn*.
#
# # In[ ]:
#
#
from sklearn.preprocessing import MultiLabelBinarizer
#
# # In[ ]:
#
#
mlb = MultiLabelBinarizer(classes=sorted(tags_counts.keys()))
y_train = mlb.fit_transform(y_train)
y_val = mlb.fit_transform(y_val)
#
# # Implement the function *train_classifier* for training a classifier.
# In this task we suggest to use One-vs-Rest approach, which is implemented in
# [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) class.
# In this approach *k* classifiers (= number of tags) are trained. As a basic classifier,
# use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
# It is one of the simplest methods, but often it performs good enough in text classification tasks.
# It might take some time, because a number of classifiers to train is large.
#
# # In[ ]:
#
#
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
#
#
# # In[ ]:
#
#
def train_classifier(X_train, y_train):
    """
      X_train, y_train — training data

      return: trained classifier
    """

    # Create and fit LogisticRegression wraped into OneVsRestClassifier.

    ######################################
    ######### YOUR CODE HERE #############
    ######################################

    return OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)

#
# # Train the classifiers for different data transformations: *bag-of-words* and *tf-idf*.
#
# # In[ ]:
#
#
classifier_mybag = train_classifier(X_train_mybag, y_train)
classifier_tfidf = train_classifier(X_train_tfidf, y_train)

# # Now you can create predictions for the data. You will need two types of predictions: labels and scores.
#
# # In[ ]:
#
#
y_val_predicted_labels_mybag = classifier_mybag.predict(X_val_mybag)
y_val_predicted_scores_mybag = classifier_mybag.decision_function(X_val_mybag)

y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)
y_val_predicted_scores_tfidf = classifier_tfidf.decision_function(X_val_tfidf)
#
# # Now take a look at how classifier, which uses TF-IDF, works for a few examples:
#
# # In[ ]:
#
#
y_val_pred_inversed = mlb.inverse_transform(y_val_predicted_labels_tfidf)
y_val_inversed = mlb.inverse_transform(y_val)
for i in range(3):
    print('Title:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
        X_val[i],
        ','.join(y_val_inversed[i]),
        ','.join(y_val_pred_inversed[i])
    ))
#
# # Now, we would need to compare the results of different predictions,
# e.g. to see whether TF-IDF transformation helps or to try different regularization techniques
# in logistic regression. For all these experiments, we need to setup evaluation procedure.
#
# # ### Evaluation
# #
# # To evaluate the results we will use several classification metrics:
# #  - [Accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
# #  - [F1-score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)
# #  - [Area under ROC-curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
# #  - [Area under precision-recall curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score)
# #
# # Make sure you are familiar with all of them. How would you expect the things work for the multi-label scenario?
# Read about micro/macro/weighted averaging following the sklearn links provided above.
#
# # In[ ]:
#
#
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
#
#
# # Implement the function *print_evaluation_scores* which calculates and prints to stdout:
# #  - *accuracy*
# #  - *F1-score macro/micro/weighted*
# #  - *Precision macro/micro/weighted*
#
# # In[ ]:
#
#
def print_evaluation_scores(y_val, predicted):

    ac_score = accuracy_score(y_val,predicted)

    f1_sc = {
        'micro' :  f1_score(y_val,predicted, average='micro'),
        'macro': f1_score(y_val, predicted, average='macro'),
        'weighted': f1_score(y_val, predicted, average='weighted')
    }

    precision_sc = {
        'micro': average_precision_score(y_val, predicted, average='micro'),
        'macro': average_precision_score(y_val, predicted, average='macro'),
        'weighted': average_precision_score(y_val, predicted, average='weighted')

    }

    return ac_score,f1_sc,precision_sc

#
# ######################################
# ######### YOUR CODE HERE #############
# ######################################
#
#
# # In[ ]:
#
#
print('Bag-of-words')
print_evaluation_scores(y_val, y_val_predicted_labels_mybag)
print('Tfidf')
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)

# # You might also want to plot some generalization of the [ROC curve](http://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc) for the case of multi-label classification. Provided function *roc_auc* can make it for you. The input parameters of this function are:
# #  - true labels
# #  - decision functions scores
# #  - number of classes
#
# # In[ ]:
#
#
from metrics import roc_auc
#
# get_ipython().magic(u'matplotlib inline')
#
# # In[ ]:
#
#
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_mybag, n_classes)
#
# # In[ ]:
#
#
n_classes = len(tags_counts)
roc_auc(y_val, y_val_predicted_scores_tfidf, n_classes)
#
# # **Task 4 (MultilabelClassification).** Once we have the evaluation set up, we suggest that you
# experiment a bit with training your classifiers. We will use *F1-score weighted* as an evaluation metric. Our recommendation:
# # - compare the quality of the bag-of-words and TF-IDF approaches and chose one of them.
# # - for the chosen one, try *L1* and *L2*-regularization techniques in Logistic Regression with different
# coefficients (e.g. C equal to 0.1, 1, 10, 100).
# #
# # You also could try other improvements of the preprocessing / model, if you want.
#
# # In[ ]:
#
#
# ######################################
# ######### YOUR CODE HERE #############
# ######################################
#
#
# # When you are happy with the quality, create predictions for *test* set, which you will submit to Coursera.
#
# # In[ ]:
#
#
# test_predictions =  ######### YOUR CODE HERE #############
# test_pred_inversed = mlb.inverse_transform(test_predictions)
#
# test_predictions_for_submission = '\n'.join('%i\t%s' % (i, ','.join(row)) for i, row in enumerate(test_pred_inversed))
# grader.submit_tag('MultilabelClassification', test_predictions_for_submission)
#
#
# # ### Analysis of the most important features
#
# # Finally, it is usually a good idea to look at the features (words or n-grams) that are used with the largest
# weigths in your logistic regression model.
#
# # Implement the function *print_words_for_tag* to find them. Get back to sklearn documentation on
# [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
# and [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) if needed.
#
# # In[ ]:
#
#
# def print_words_for_tag(classifier, tag, tags_classes, index_to_words, all_words):
#     """
#         classifier: trained classifier
#         tag: particular tag
#         tags_classes: a list of classes names from MultiLabelBinarizer
#         index_to_words: index_to_words transformation
#         all_words: all words in the dictionary
#
#         return nothing, just print top 5 positive and top 5 negative words for current tag
#     """
#     print('Tag:\t{}'.format(tag))
#
#     # Extract an estimator from the classifier for the given tag.
#     # Extract feature coefficients from the estimator.
#
#     ######################################
#     ######### YOUR CODE HERE #############
#     ######################################
#
#     top_positive_words =  # top-5 words sorted by the coefficiens.
#     top_negative_words =  # bottom-5 words  sorted by the coefficients.
#     print('Top positive words:\t{}'.format(', '.join(top_positive_words)))
#     print('Top negative words:\t{}\n'.format(', '.join(top_negative_words)))
#
#
# # In[ ]:
#
#
# print_words_for_tag(classifier_tfidf, 'c', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
# print_words_for_tag(classifier_tfidf, 'c++', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
# print_words_for_tag(classifier_tfidf, 'linux', mlb.classes, tfidf_reversed_vocab, ALL_WORDS)
#
# # ### Authorization & Submission
# # To submit assignment parts to Cousera platform, please, enter your e-mail and token into variables below. You can generate token on this programming assignment page. <b>Note:</b> Token expires 30 minutes after generation.
#
# # In[ ]:
#
#
# grader.status()
#
# # In[ ]:
#
#
# STUDENT_EMAIL =  # EMAIL
# STUDENT_TOKEN =  # TOKEN
# grader.status()
#
# # If you want to submit these answers, run cell below
#
# # In[ ]:
#
#
# grader.submit(STUDENT_EMAIL, STUDENT_TOKEN)
