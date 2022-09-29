import joblib
def get_prediction(tfidf_out):
    x_input = tfidf_out
    print(x_input)
    nb = joblib.load('P://IIT_spl3BackEnd_ML//saved_models//NB.pkl')
    #P:\IIT_spl3BackEnd_ML\saved_models\NB.pkl'''
    y_preds = nb.predict(x_input)
    # report = classification_report(y_test,y_preds)
    print(y_preds)
    print(y_preds.size)
    return y_preds