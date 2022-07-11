from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('search', views.search, name='search'),
    path('error', views.error, name='error'),
]
