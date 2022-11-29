import pandas as panda
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import seaborn
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VS
import warnings
import pickle

from joblib import Parallel, delayed
import joblib


# warnings.simplefilter(action='ignore', category=FutureWarning)
# %matplotlib inline


def read_csv():
    dataset = panda.read_csv('P:\Spl3\IIT_spl3BackEnd_ML\profanity_check\HateSpeechData.csv')
    # Adding text-length as a field in the dataset
    dataset['text length'] = dataset['tweet'].apply(len)
    return dataset


def get_tweets():
    dataset = read_csv()
    # collecting only the tweets from the csv file into a variable name tweet
    return dataset.tweet
