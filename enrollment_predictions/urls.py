from django.urls import path

from . import views

urlpatterns = [
    path('', views.predict_endpoint, name='predict'),
]


