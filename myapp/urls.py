from django.urls import path

from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('predict-detailed/', views.predict_detailed, name='predict-detailed'),
    path('train/', views.train, name='train'),
    path('train-detailed/', views.train_detailed, name='train-detailed'),
]


