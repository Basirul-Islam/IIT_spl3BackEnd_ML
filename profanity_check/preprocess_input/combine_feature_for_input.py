from get_comments.get_video_comments import *
from preprocess_input_data.preprocess_comments import *
from read_and_process_data.preprocess_data import *
from profanity_check.preprocess_input.extract_feature_for_input_data import *
def combine_feacture_for_input():
    path_url = "https://www.youtube.com/watch?v=jEdfjuG0Fx4"
    input_comments_with_id = get__comments(path_url=path_url)
    processed_input_comments = process_comments(input_comments_with_id)

    #preprocess_input_comments = preprocess(processed_input_comments)
    #print(preprocess_input_comments)
    tfidf = get_tfidf_of_input_comments(processed_input_comments)
    tfidf_a = tfidf.toarray()
    doc2vec_df = get_doc2vec_for_input_data(comments=processed_input_comments)

    print(doc2vec_df)

    fFeatures = get_additional_feature_for_input_comments(processed_input_comments)
    modelling_features_three = np.concatenate([tfidf_a, doc2vec_df, fFeatures], axis=1)
    modelling_features_three.shape
    return modelling_features_three

    #return preprocess_input_comments