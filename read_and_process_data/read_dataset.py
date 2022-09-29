import pandas as pd
from read_and_process_data.preprocess_data import *
def read_data():
    dataset = pd.read_csv("read_and_process_data/YoutubeSpamMergedData.csv")
    dataset = dataset[["CONTENT", "CLASS"]]
    dataset['text length'] = dataset['CONTENT'].apply(len)
    comments = dataset.CONTENT
    #print(comments)
    #dataset['processed_data'] = preprocess(comments)
    processed_comments = preprocess(comments)
    dataset['processed_comments'] = processed_comments
    #print(dataset[["processed_comments"]].head(10))
    #get_tfidf(dataset=dataset)
    return dataset
    #print(dataset.head(10))

def get_tfidf(dataset):
    #dataset = read_data()
    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF Features-F1
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['processed_comments'])
    tfidf
    #print(tfidf)
    return tfidf

def get_input_comments_tfidf(df):
    dataset = read_data()
    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF Features-F1
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['processed_comments'])
    tfidf
    tfidf_out = tfidf_vectorizer.transform(df['processed_comments'])
    tfidf_out
    #print(tfidf)
    return tfidf_out