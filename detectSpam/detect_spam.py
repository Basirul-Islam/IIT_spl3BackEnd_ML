import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

class comment_with_prediction:
    def __init__(self, comment, prediction, ):
        self.comment = comment
        self.prediction = prediction


'''class accuracy_of_model:
    def __init__(self, model_name, accuracy, ):
        self.model_name = model_name
        self.accuracy = accuracy


def get_accuracy():
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

    # Load the model from the file
    nb = joblib.load('pickles/nb.pkl')
    logisticreg = joblib.load('pickles/LogisticReg.pkl')
    dtc = joblib.load('pickles/dtc.pkl')
    rfc = joblib.load('pickles/rfc.pkl')
    svc = joblib.load('pickles/svc.pkl')

    nb_accuracy = nb.score(X_test, y_test)*100
    logisticreg_accuracy = logisticreg.score(X_test, y_test)*100
    dtc_accuracy = dtc.score(X_test, y_test)*100
    rfc_accuracy = rfc.score(X_test, y_test)*100
    svc_accuracy = svc.score(X_test, y_test)*100

    accuracy_arr = []
    accuracy_arr.append(accuracy_of_model("Naive Bayes", nb_accuracy).__dict__)
    accuracy_arr.append(accuracy_of_model("Logistic Regression", logisticreg_accuracy).__dict__)
    accuracy_arr.append(accuracy_of_model("Decision Tree", dtc_accuracy).__dict__)
    accuracy_arr.append(accuracy_of_model("Random Forest", rfc_accuracy).__dict__)
    accuracy_arr.append(accuracy_of_model("Support Vector Machine", svc_accuracy).__dict__)

    return accuracy_arr'''

def predict(comment):
    '''df = pd.read_csv("detectSpam/YoutubeSpamMergedData.csv")
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
    clf.fit(X_train, y_train)'''

    # clf.score(X_test, y_test)
    '''for item in df.head():
        data = [item]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
        print(prediction)'''

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
    # Load the model from the file
    logistic_reg = joblib.load('pickles/LogisticReg.pkl')
    data = [comment]
    vect = cv.transform(data).toarray()
    my_prediction = logistic_reg.predict(vect)
    print(data)
    return my_prediction
    # obj = comment_with_prediction(comment, my_prediction)
    # return obj.__dict__
