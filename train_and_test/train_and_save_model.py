import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from read_and_process_data.read_dataset import *
def train_and_save_model():
    dataset = read_data()
    tfidf = get_tfidf(dataset = dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = MultinomialNB()
    lr = LogisticRegression()
    svc = SVC()
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    nb.fit(X_train_tfidf, y_train)
    lr.fit(X_train_tfidf, y_train)
    svc.fit(X_train_tfidf, y_train)
    rf.fit(X_train_tfidf, y_train)
    dt.fit(X_train_tfidf, y_train)
    joblib.dump(nb, 'P://Projects//IIT_spl3BackEnd_ML//saved_models//NB.pkl')
    joblib.dump(lr, 'P://Projects//IIT_spl3BackEnd_ML//saved_models//LR.pkl')
    joblib.dump(svc, 'P://Projects//IIT_spl3BackEnd_ML//saved_models//SVC.pkl')
    joblib.dump(rf, 'P://Projects//IIT_spl3BackEnd_ML//saved_models//RF.pkl')
    joblib.dump(dt, 'P://Projects//IIT_spl3BackEnd_ML//saved_models//DT.pkl')
    return ("ok")

def get_model_report():
    report = []
    dataset = read_data()
    tfidf = get_tfidf(dataset=dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = joblib.load('P://Projects//IIT_spl3BackEnd_ML//saved_models//NB.pkl')
    lr = joblib.load('P://Projects//IIT_spl3BackEnd_ML//saved_models//LR.pkl')
    svc = joblib.load('P://Projects//IIT_spl3BackEnd_ML//saved_models//SVC.pkl')
    rf = joblib.load('P://Projects//IIT_spl3BackEnd_ML//saved_models//RF.pkl')
    dt = joblib.load('P://Projects//IIT_spl3BackEnd_ML//saved_models//DT.pkl')
    y_preds = nb.predict(X_test_tfidf)
    model_report = classification_report(y_test, y_preds)
    report.append(model_report)
    y_preds = lr.predict(X_test_tfidf)
    model_report = classification_report(y_test, y_preds)
    report.append(model_report)
    #print(report)
    return report