from rest_framework.routers import DefaultRouter
from django.urls import include, path
from .views import *

router = DefaultRouter()

urlpatterns = [
    path('inference', InferenceCreateAPIView.as_view())
]