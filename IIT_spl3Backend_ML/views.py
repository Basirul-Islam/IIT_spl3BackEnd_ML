from rest_framework.decorators import api_view
from rest_framework.response import Response
from getComments.extract_comments import *
from detectSpam.detect_spam import *


@api_view(['GET'])
def hellow(request):
    print("triggered")
    return Response(data={"hellow, welcome to stocker"})


@api_view(['GET'])
def video_comments(request, url):
    # print(url)
    print(url)
    data = get__comments()
    return Response(data=data)


@api_view(['GET'])
def detect_spam(request, comment):
    print("comment: ", comment)
    prediction = predict(comment)
    return Response(data=prediction)
