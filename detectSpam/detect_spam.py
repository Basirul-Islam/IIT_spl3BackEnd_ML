import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


class comment_with_prediction:
    def __init__(self, comment, prediction, ):
        self.comment = comment
        self.prediction = prediction


def predict(comment):
    df = pd.read_csv("detectSpam/YoutubeSpamMergedData.csv")
    # print(df.head())
    df_data = df[["CONTENT", "CLASS"]]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)  # Fit the Data

    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # clf.score(X_test, y_test)
    '''for item in df.head():
        data = [item]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
        print(prediction)'''
    data = [comment]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)
    return my_prediction
    #obj = comment_with_prediction(comment, my_prediction)
    #return obj.__dict__
