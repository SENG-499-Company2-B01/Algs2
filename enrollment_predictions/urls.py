from django.urls import path

from . import views

urlpatterns = [
    path('generate/', views.generate, name='generate'),
    path('notify/', views.notify, name='notify'),
]


