from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

# Create your models here.
class Picture(models.Model):

    photo_before = models.ImageField(upload_to='photos/%Y/%m/%d/', verbose_name='Фото оригінал')
    photo_after = models.ImageField(upload_to='photos/%Y/%m/%d/', verbose_name='Фото колоризоване')
    time_create = models.DateTimeField(auto_now_add=True, verbose_name='Час створення')
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True)


    def __str__(self):
        return f"{self.photo_before}"

    # def get_absolute_url(self):
    #     return reverse('personal_cabinet', kwargs={'pk': self.pk})

    # def get_absolute_url(self):
    #     return reverse('post', kwargs={'post_slug':self.slug})

    # class Meta:
    #     verbose_name = 'Відомі жінки'
    #     verbose_name_plural = 'Відомі жінки'
    #     ordering = ['-time_create', 'title']