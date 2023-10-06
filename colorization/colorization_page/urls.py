from django.urls import path
from .views import *

urlpatterns = [
    path('', index, name='index'),
    path("login", login_view, name="login"),
    path("logout", logout_view, name="logout"),
    path("register", register, name="register"),
    path('personal_cabinet/<int:pk>/', personal_cabinet, name='personal_cabinet'),
    path('picture/<int:pk>/delete/', delete_picture, name='delete_picture'),

]