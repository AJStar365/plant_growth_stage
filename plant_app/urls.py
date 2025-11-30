from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path("", auth_views.LoginView.as_view(), name="login"),
    path("predict/", views.index, name="index"),
    path("signup/", views.signup, name="signup"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
