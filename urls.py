# urls.py

from django.urls import path
from . import app

urlpatterns = [
    # URL pattern for the homepage view
    path('', app.homepage, name='homepage'),

    # URL pattern for the predict_gender_route view with a dynamic parameter 'noun'
    path('predict_gender/<str:noun>/', app.predict_gender_route, name='predict_gender'),
]
