from rest_framework.decorators import api_view
from rest_framework.response import Response
from accuracy.model_accuracy import *
from get_comments.get_spam_labeled_comments import *
from train_and_test.train_and_save_model import *
@api_view(['GET'])
def hellow(request):
    print("triggered")
    return Response(data={"hellow, welcome by Bashir"})

@api_view(['GET'])
def spam_comments(request):
    data = get_spam_comments()
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