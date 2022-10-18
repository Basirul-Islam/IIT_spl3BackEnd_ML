import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from read_and_process_data.read_dataset import *
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics

class model_report:
    def __init__(self, model_name, accuracy, f1Score, precision, recall, confusionMetrics):
        self.model_name = model_name
        self.accuracy = accuracy
        self.f1Score = f1Score
        self.precision = precision
        self.recall = recall
        self.confusionMetrics = confusionMetrics


def train_and_save_model():
    dataset = read_data()
    tfidf = get_tfidf(dataset=dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = MultinomialNB()
    lr = LogisticRegression()
    svc = SVC(probability=True)
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    nb.fit(X_train_tfidf, y_train)
    lr.fit(X_train_tfidf, y_train)
    svc.fit(X_train_tfidf, y_train)
    rf.fit(X_train_tfidf, y_train)
    dt.fit(X_train_tfidf, y_train)
    joblib.dump(nb, 'saved_models/NB.pkl')
    joblib.dump(lr, 'saved_models/LR.pkl')
    joblib.dump(svc, 'saved_models/SVC.pkl')
    joblib.dump(rf, 'saved_models/RF.pkl')
    joblib.dump(dt, 'saved_models/DT.pkl')
    return ("ok")

def get_metric(model_name, y_test, y_pred):
    confusion_metric = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    accuracy = accuracy_score(y_test, y_pred)
    #print("confusion_metrix: ", confusion_metric, "f1_score: ", f1_score(y_test, y_pred))
    report = model_report(model_name, accuracy, f1, precision, recall, confusion_metric)
    print(report.__dict__)
    return report.__dict__
def get_model_report():
    report = []
    dataset = read_data()
    tfidf = get_tfidf(dataset=dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = joblib.load('saved_models/NB.pkl')
    lr = joblib.load('saved_models/LR.pkl')
    svc = joblib.load('saved_models/SVC.pkl')
    rf = joblib.load('saved_models/RF.pkl')
    dt = joblib.load('saved_models/DT.pkl')
    #y_preds = nb.predict(X_test_tfidf)
    #model_accuracy = accuracy_score(y_test, y_preds)
    #model_report("Naive Bayes", model_accuracy)

    nb_y_pred = (nb.predict_proba(X_test_tfidf)[:, 1] >= 0.8).astype(int)
    lr_y_pred = (lr.predict_proba(X_test_tfidf)[:, 1] >= 0.8).astype(int)
    svc_y_pred = (svc.predict_proba(X_test_tfidf)[:, 1] >= 0.8).astype(int)
    rf_y_pred = (rf.predict_proba(X_test_tfidf)[:, 1] >= 0.8).astype(int)
    dt_y_pred = (dt.predict_proba(X_test_tfidf)[:, 1] >= 0.8).astype(int)
    report.append(get_metric("Naive Bayes", y_test, nb_y_pred))
    report.append(get_metric("Logistic Regression", y_test, lr_y_pred))
    report.append(get_metric("Support Vector Classifier", y_test, svc_y_pred))
    report.append(get_metric("Random Forest Classifier", y_test, rf_y_pred))
    report.append(get_metric("Decision Tree Classifier", y_test, dt_y_pred))

    #model_accuracy = accuracy_score(y_test, nb_y_pred)
    #model_report("Naive Bayes", model_accuracy)
    #nb_confusion_metric = metrics.confusion_matrix(y_test, nb_y_pred)

    #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=nb_confusion_metric, display_labels=["Spam", "Ham"])
    #cm_display.plot()
    #plt.show()

    #get_metric("NB", y_test, nb_y_pred)
    #print(nb_confusion_metric)
    '''print("confusion metric", confusion_matrix(y_test, nb_y_pred))
    print(f1_score(y_test, nb_y_pred, average="macro"))
    print(precision_score(y_test, nb_y_pred, average="macro"))
    print(recall_score(y_test, nb_y_pred, average="macro"))'''




    '''report.append(model_report("Naive Bayes", model_accuracy).__dict__)
    y_preds = lr.predict(X_test_tfidf)
    model_accuracy = accuracy_score(y_test, y_preds)
    report.append(model_report("Logistic Regression", model_accuracy).__dict__)
    y_preds = svc.predict(X_test_tfidf)
    model_accuracy = accuracy_score(y_test, y_preds)
    report.append(model_report("Support Vector Classifier", model_accuracy).__dict__)
    y_preds = rf.predict(X_test_tfidf)
    model_accuracy = accuracy_score(y_test, y_preds)
    report.append(model_report("Random Forest Classifier", model_accuracy).__dict__)
    y_preds = dt.predict(X_test_tfidf)
    model_accuracy = accuracy_score(y_test, y_preds)
    report.append(model_report("Decision Tree Classifier", model_accuracy).__dict__)'''

    # model_report = classification_report(y_test, y_preds)
    # report.append(model_report)
    y_preds = lr.predict(X_test_tfidf)
    # model_report = classification_report(y_test, y_preds)
    # report.append(model_report)
    # print(report)
    return report
