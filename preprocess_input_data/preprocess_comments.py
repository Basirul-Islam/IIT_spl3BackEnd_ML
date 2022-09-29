from get_comments.get_video_comments import *
from read_and_process_data.read_dataset import *
from train_and_test.prediction import *
import pandas as pd
def process_comments(input_comments_with_id):
    #input_comments_with_id = get__comments()
    input_comments = []
    # input_comments_json = json.loads(input_comments)
    for input_comment in input_comments_with_id:
        input_comments.append(input_comment['comment'])
        # print(input_comment['comment'])
    print(input_comments)
    return input_comments
def get_tfidf_of_input_comments(processed_input_comments):
    #input_comments = process_comments()
    df = pd.DataFrame(processed_input_comments, columns=['processed_comments'])
    print(df[["processed_comments"]])
    tfidf_out = get_input_comments_tfidf(df=df)
    return tfidf_out

    #print(get_tfidf(dataset=df))
