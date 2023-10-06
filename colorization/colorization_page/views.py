from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from .main import *
from django.contrib.auth.models import User
from .forms import *
from django.shortcuts import get_object_or_404, redirect
import os

def index(request):
    if request.method == 'POST':
        form = PictureForm(request.POST, request.FILES)
        if form.is_valid():
            picture = form.save()
            if request.user.is_authenticated:
                picture.author = request.user
            best_model_weights_path = "colorization_page/cnn_model_last.h5"
            best_model = cnn_model()
            best_model.load_weights(best_model_weights_path)
            colorized_photo = plot_result(picture.photo_before.url[1:], best_model)

            picture.photo_after = '/'.join(colorized_photo.split('/')[1:])

            picture.save()
            photo = Picture.objects.last()
        else:
            form = PictureForm()
            photo = None

    else:
        form = PictureForm()
        photo = None


    context = {'form': form, 'photo_after': photo}
    return render(request, 'colorization_page/index.html', context)

def personal_cabinet(request, pk):
    projects = Picture.objects.filter(author_id=pk).order_by('-time_create')
    return render(request, 'colorization_page/personal_cabinet.html', {
        'projects': projects
    })


def delete_picture(request, pk):
    picture = get_object_or_404(Picture, pk=pk)
    if request.method == 'POST':
        picture.delete()
        os.remove(picture.photo_before.path)
        os.remove(picture.photo_after.path)
        return redirect('personal_cabinet', request.user.pk)
    return HttpResponseRedirect(reverse("index"))



def login_view(request):
    if request.method == "POST":
        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "colorization_page/login.html", {
                "message": "Неправильне ім'я користувача та/або пароль."
            })
    else:
        return render(request, "colorization_page/login.html")


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "colorization_page/register.html", {
                "message": "Паролі повинні збігатися."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "colorization_page/register.html", {
                "message": "Ім'я користувача вже зайнято."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "colorization_page/register.html")

