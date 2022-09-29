import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from read_and_process_data.read_dataset import *
def train_and_save_model():
    dataset = read_data()
    tfidf = get_tfidf(dataset = dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)
    joblib.dump(nb, 'C://Users//cefalo//PycharmProjects//test//spl3//demo//saved_models//NB.pkl')

def get_model_report():
    dataset = read_data()
    tfidf = get_tfidf(dataset=dataset)
    X = tfidf
    y = dataset['CLASS'].astype(int)
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    nb = joblib.load('C://Users//cefalo//PycharmProjects//test//spl3//demo//saved_models//NB.pkl')
    y_preds = nb.predict(X_test_tfidf)
    report = classification_report(y_test, y_preds)
    print(report)
    return report