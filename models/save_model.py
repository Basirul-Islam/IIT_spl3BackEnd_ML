import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_model_and_save():
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

    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    joblib.dump(reg, 'LogisticReg.pkl')

    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    joblib.dump(dtc, 'dtc.pkl')

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, 'nb.pkl')

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    joblib.dump(rfc, 'rfc.pkl')

    svc = SVC()
    svc.fit(X_train, y_train)
    joblib.dump(svc, 'svc.pkl')