from sklearn.calibration import CalibratedClassifierCV

from profanity_check.combine_feature import *
from profanity_check.read_dataset import *
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_model_and_save():
    dataset = read_csv()
    modelling_features_two = get_combine_feature_3()
    X = panda.DataFrame(modelling_features_two)
    y = dataset['class'].astype(int)
    X_train_features, X_test_features, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
    support = LinearSVC(random_state=20)
    #clf = CalibratedClassifierCV(support)
    #support = SVC(probability=True)
    #nb = MultinomialNB()
    #nb.fit(X_train_features, y_train)
    #joblib.dump(nb, 'P:/Spl3/IIT_spl3BackEnd_ML/profanity_check/saved_models/NB.pkl')
    support.fit(X_train_features, y_train)

    joblib.dump(support, 'P:/Spl3/IIT_spl3BackEnd_ML/profanity_check/saved_models/SVC.pkl')

    y_preds = support.predict(X_test_features)
    acc3 = accuracy_score(y_test, y_preds)
    report = classification_report(y_test, y_preds)
    print(report)
    print("SVM, Accuracy Score:", acc3)
    return report
