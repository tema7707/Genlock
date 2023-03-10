import os

from django.urls import path
from .views import TrainModelView, GenereateView

urlpatterns = [
    path('api/v1/train', TrainModelView.as_view(), name='train'),
    path('api/v1/generate/<str:asset_type>/', GenereateView.as_view(), name='generate'),
]
