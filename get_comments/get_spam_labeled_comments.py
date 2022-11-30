from get_comments.get_video_comments import *
from preprocess_input_data.preprocess_comments import *
from train_and_test.prediction import *
from profanity_check.test_model.test_saved_model import *


class comments_with_spam_pred:
    def __init__(self, id, comment, authorDisplayName, profileImageUrl, pred):
        self.id = id
        self.comment = comment
        self.authorDisplayName = authorDisplayName
        self.profileImageUrl = profileImageUrl
        self.prediction = pred
        '''if pred == 1:
            self.prediction = 1
        else: self.prediction = 0'''


class comments_with_spam_hate_pred:
    def __init__(self, id, comment, authorDisplayName, profileImageUrl, spam_pred, hate_pred, combine_label):
        self.id = id
        self.comment = comment
        self.authorDisplayName = authorDisplayName
        self.profileImageUrl = profileImageUrl
        self.spam_prediction = spam_pred
        self.hate_prediction = hate_pred
        self.combine_label = combine_label


def get_spam_comments(path_url):
    input_comments_with_id = get__comments(path_url=path_url)
    processed_input_comments = process_comments(input_comments_with_id)
    tfidf_out = get_tfidf_of_input_comments(processed_input_comments)
    y_preds = get_prediction(tfidf_out=tfidf_out)
    count = 0
    comments_with_spam_prediction = []
    for input_comment in input_comments_with_id:
        comment_isSpam = comments_with_spam_pred(input_comment['id'], input_comment['comment'],
                                                 input_comment['authorDisplayName'], input_comment['profileImageUrl'],
                                                 y_preds[count])
        count = count + 1
        comments_with_spam_prediction.append(comment_isSpam.__dict__)
    return comments_with_spam_prediction


def get_spam_and_hate_comments(path_url):
    print(path_url)
    input_comments_with_id = get__comments(path_url=path_url)
    processed_input_comments = process_comments(input_comments_with_id)
    #print(processed_input_comments)
    tfidf_out = get_tfidf_of_input_comments(processed_input_comments)
    y_preds = get_prediction(tfidf_out=tfidf_out)
    #print(y_preds)
    y_preds_for_hate = get_profanity_prediction(path_url=path_url)
    count = 0
    comments_with_spam__and_hate_prediction = []
    #combine_label = 0
    # spam predict = 0 means not spam and hate predict = 0 means hate
    for input_comment in input_comments_with_id:
        if y_preds[count] == 1 or y_preds_for_hate[count] == 0:
            combine_label = 0
        else:
            combine_label = 1
        comment_is_spam_or_hate = comments_with_spam_hate_pred(input_comment['id'], input_comment['comment'],
                                                 input_comment['authorDisplayName'], input_comment['profileImageUrl'],
                                                 y_preds[count], y_preds_for_hate[count], combine_label)
        count = count + 1
        comments_with_spam__and_hate_prediction.append(comment_is_spam_or_hate.__dict__)
    return comments_with_spam__and_hate_prediction
    #return path_url
