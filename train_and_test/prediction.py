import joblib
def get_prediction(tfidf_out):
    x_input = tfidf_out
    nb = joblib.load('saved_models/NB.pkl')
    y_preds = nb.predict(x_input)
    # report = classification_report(y_test,y_preds)
    #print(y_preds)
    #print(y_preds.size)
    return y_preds