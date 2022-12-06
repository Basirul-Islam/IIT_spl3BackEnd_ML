from profanity_check.extract_feature import *
from profanity_check.read_dataset import *


def get_combine_feature_1(dataset):
    # F2-Conctaenation of tf-idf scores and sentiment scores
    tfidf = get_tfidf(dataset=dataset)
    tfidf_a = tfidf.toarray()
    tweet = get_tfidf(dataset=dataset)
    final_features = get_primary_feature(tweet=tweet)
    modelling_features = np.concatenate([tfidf_a, final_features], axis=1)
    modelling_features.shape


def get_combine_feature_2():
    dataset = read_csv()
    #conctaenation of tf-idf scores, sentiment scores and doc2vec columns
    tfidf = get_tfidf(dataset=dataset)
    tfidf_a = tfidf.toarray()
    #tweet = get_tweets(dataset=dataset)
    final_features = get_primary_feature(tweet=dataset.tweet)
    doc2vec_df = get_doc2vec(dataset=dataset)
    print(doc2vec_df)
    modelling_features = np.concatenate([tfidf_a, final_features, doc2vec_df], axis=1)
    modelling_features.shape
    #print(get_doc2vec())
    #return doc2vec_df
    return modelling_features

def get_combine_feature_3():
    # f1,f3 and f4 combined

    dataset = read_csv()
    # conctaenation of tf-idf scores, sentiment scores and doc2vec columns
    tfidf = get_tfidf(dataset=dataset)
    tfidf_a = tfidf.toarray()
    doc2vec_df = get_doc2vec(dataset=dataset)
    print(doc2vec_df)
    fFeatures = get_additional_feature(dataset.tweet)
    print(fFeatures)
    modelling_features_three = np.concatenate([tfidf_a, doc2vec_df, fFeatures], axis=1)
    modelling_features_three.shape
    return modelling_features_three

def get_combine_feature_4():
    # f1,f2 and f4 combined
    dataset = read_csv()
    # conctaenation of tf-idf scores, sentiment scores and doc2vec columns
    tfidf = get_tfidf(dataset=dataset)
    tfidf_a = tfidf.toarray()
    fFeatures = get_additional_feature(dataset.tweet)
    final_features = get_primary_feature(tweet=dataset.tweet)
    modelling_features_four = np.concatenate([tfidf_a, final_features, fFeatures], axis=1)
    modelling_features_four.shape
    return modelling_features_four

