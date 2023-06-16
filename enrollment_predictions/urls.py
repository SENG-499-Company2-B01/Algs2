from django.urls import path

from . import views

urlpatterns = [
    path('predict/', views.my_endpoint, name='predict'),
]


