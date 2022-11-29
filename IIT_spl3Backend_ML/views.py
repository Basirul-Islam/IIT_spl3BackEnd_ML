from rest_framework.decorators import api_view
from rest_framework.response import Response
from accuracy.model_accuracy import *
from get_comments.get_spam_labeled_comments import *
from get_comments.get_video_comments import *
from train_and_test.train_and_save_model import *
from data_models.url import *
from profanity_check.combine_feature import *
from profanity_check.train_and_save_model import *
from profanity_check.preprocess_input.combine_feature_for_input import *
from profanity_check.test_model.test_saved_model import *
import json
@api_view(['POST'])
def hellow(request):
    #serializers.serialize('json', self.get_queryset())
    #url = url_path(request.data).data.values()
    for url in url_path(request.data).data.values():
        print(url)
        break
    #return Response(data=json.dumps(url))
    return Response(data={"hellow, welcome by Bashir"})

@api_view(['Post'])
def spam_comments(request):
    for url in url_path(request.data).data.values():
        path_url = url
        #print(url)
        break
    data = get_spam_comments(path_url)
    return Response(data=data)

@api_view(['Post'])
def get_comments(request):
    for url in url_path(request.data).data.values():
        path_url = url
        #print(url)
        break
    data = get__comments(path_url)
    return Response(data=data)

@api_view(['GET'])
def save_model(request):
    data = train_and_save_model()
    return Response(data=data)

'''@api_view(['GET'])
def video_comments(request, url):
    # print(url)
    print(url)
    data = get__comments()
    return Response(data=data)'''


'''@api_view(['GET'])
def detect_spam(request, comment):
    print("comment: ", comment)
    prediction = predict(comment)
    return Response(data=prediction)'''

@api_view(['GET'])
def model_accuracy(request):
    data = get_model_report()
    return Response(data=data)

@api_view(['GET'])
def get_doc2vec(request):
    #data = get_combine_feature_2()
    #data = train_model_and_save()
    #data = combine_feacture_for_input()
    data = get_profanity_prediction()
    return Response(data=data)