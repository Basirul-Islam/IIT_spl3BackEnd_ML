from profanity_check.preprocess_comments import *
from read_and_process_data.preprocess_data import *
from profanity_check.read_dataset import *
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_tfidf(dataset):
    # dataset = read_data()

    tweets = dataset.tweet
    processed_comments = preprocess(tweets)
    dataset['processed_tweets'] = processed_comments

    from sklearn.feature_extraction.text import TfidfVectorizer

    # TF-IDF Features-F1
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
    # TF-IDF feature matrix
    tfidf = tfidf_vectorizer.fit_transform(dataset['processed_tweets'])
    tfidf
    # print(tfidf)
    return tfidf


def sentiment_analysis(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    twitter_objs = count_tags(tweet)
    features = [sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'], twitter_objs[0],
                twitter_objs[1],
                twitter_objs[2]]
    # features = pandas.DataFrame(features)
    return features


def sentiment_analysis_array(tweets):
    features = []
    for t in tweets:
        features.append(sentiment_analysis(t))
    return np.array(features)


def get_primary_feature(tweet):
    # print(tweet)
    final_features = sentiment_analysis_array(tweet)
    # final_features

    new_features = panda.DataFrame(
        {'Neg': final_features[:, 0], 'Pos': final_features[:, 1], 'Neu': final_features[:, 2],
         'Compound': final_features[:, 3],
         'url_tag': final_features[:, 4], 'mention_tag': final_features[:, 5], 'hash_tag': final_features[:, 6]})
    return new_features


def get_doc2vec(dataset):
    # create doc2vec vector columns
    # Initialize and train the model
    # from gensim.test.utils import common_texts

    # The input for a Doc2Vec model should be a list of TaggedDocument(['list','of','word'], [TAG_001]).
    # A good practice is using the indexes of sentences as the tags.
    # dataset = read_csv()
    # dataset = panda.read_csv('P:\Spl3\IIT_spl3BackEnd_ML\profanity_check\HateSpeechData.csv')
    # # Adding text-length as a field in the dataset
    # dataset['text length'] = dataset['tweet'].apply(len)
    tweets = dataset.tweet
    processed_comments = preprocess(tweets)
    dataset['processed_tweets'] = processed_comments
    documents = [TaggedDocument(doc, [i]) for i, doc in
                 enumerate(dataset["processed_tweets"].apply(lambda x: x.split(" ")))]

    # train a Doc2Vec model with our text data
    # window- The maximum distance between the current and predicted word within a sentence.
    # mincount-Ignores all words with total frequency lower than this.
    # workers -Use these many worker threads to train the model
    #  Training Model - distributed bag of words (PV-DBOW) is employed.
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    # infer_vector - Infer a vector for given post-bulk training document.
    # Syntax- infer_vector(doc_words, alpha=None, min_alpha=None, epochs=None, steps=None)
    # doc_words-A document for which the vector representation will be inferred.

    # transform each document into a vector data
    doc2vec_df = dataset["processed_tweets"].apply(lambda x: model.infer_vector(x.split(" "))).apply(panda.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    return doc2vec_df


# Using TFIDF with sentiment scores,doc2vec and enhanced features
def additional_features(tweet):
    syllables = textstat.syllable_count(tweet)
    num_chars = sum(len(w) for w in tweet)
    num_chars_total = len(tweet)
    num_words = len(tweet.split())
    # avg_syl = total syllables/ total words
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(tweet.split()))

    #  Flesch–Kincaid readability tests are readability tests
    #  designed to indicate how difficult a passage in English is to understand.
    # There are two tests, the Flesch Reading Ease, and the Flesch–Kincaid Grade
    # A text with a comparatively high score on FRE test should have a lower score on the FKRA test.
    # Reference - https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

    ###Modified FK grade, where avg words per sentence is : just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    add_features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_words,
                    num_unique_terms]
    return add_features


def get_additonal_feature_array(tweets):
    features = []
    for t in tweets:
        features.append(additional_features(t))
    return np.array(features)


def get_additional_feature(tweets):
    processed_tweets = preprocess(tweets)
    fFeatures = get_additonal_feature_array(processed_tweets)
    return fFeatures
