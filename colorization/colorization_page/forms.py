from django import forms
from .models import Picture

class PictureForm(forms.ModelForm):
    class Meta:
        model = Picture
        fields = ['photo_before']
        labels = {
            'photo_before': 'Фото оригінал',
        }
        widgets = {
            'photo_before': forms.FileInput(attrs={'class': 'form-control', 'id': 'image-selector'}),
        }