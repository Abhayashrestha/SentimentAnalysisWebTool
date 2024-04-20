from django.urls import path
from . import views

app_name = 'webApp'
urlpatterns = [
    path('sentiment_analysis/', views.sentiment_analysis, name='sentiment_analysis'),
    path('news/', views.news, name='news'),
    # other URL patterns
]