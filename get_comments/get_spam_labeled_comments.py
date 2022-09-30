from get_comments.get_video_comments import *
from preprocess_input_data.preprocess_comments import *
from train_and_test.prediction import *
class comments_with_spam_pred:
    def __init__(self, id, comment, pred):
        self.id = id
        self.comment = comment
        if pred == 1:
            self.prediction = "spam"
        else: self.prediction = "not spam"
def get_spam_comments(path_url):
    input_comments_with_id = get__comments(path_url=path_url)
    processed_input_comments = process_comments(input_comments_with_id)
    tfidf_out = get_tfidf_of_input_comments(processed_input_comments)
    y_preds = get_prediction(tfidf_out=tfidf_out)
    count = 0;
    comments_with_spam_prediction = []
    for input_comment in input_comments_with_id:
        comment_isSpam = comments_with_spam_pred(input_comment['id'], input_comment['comment'], y_preds[count])
        count = count + 1
        comments_with_spam_prediction.append(comment_isSpam.__dict__)
    return comments_with_spam_prediction