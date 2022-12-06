"""TestdjangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .views import *
urlpatterns = [
    #path('admin/', admin.site.urls),
    path('hellow/', hellow),
    path('spam_comments/', spam_comments),
    path('get_comments/', get_comments),
    path('save_model/', save_model),
    #path('video_comments/(?P<url>\w+)/', video_comments),
    #path('video_comments/<str:url>/', video_comments),
    #path('detect_spam/<str:comment>/', detect_spam),
    path('get_report/', model_accuracy),
    path('get_spam_hate_comments/', get_spam_hate_comments),
    #path('get_doc2vec/', get_doc2vec),
    #path('video_comments/<path converter: URL parmeter name>/', video_comments),
]
