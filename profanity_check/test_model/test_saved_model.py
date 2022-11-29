import joblib
from profanity_check.preprocess_input.combine_feature_for_input import *

def get_profanity_prediction():
    x_input = combine_feacture_for_input()
    nb = joblib.load('P:/Spl3/IIT_spl3BackEnd_ML/profanity_check/saved_models/SVC.pkl')
    #dt = joblib.load('saved_models/DT.pkl')
    # thresholding probability
    #y_preds = (nb.predict_proba(x_input)[:, 1] >= 0.6).astype(int)

    y_preds = nb.predict(x_input)
    #probability = nb.predict_proba(x_input)
    # report = classification_report(y_test,y_preds)
    print(y_preds)
    #print(probability)
    # print(y_preds.size)
    return y_preds