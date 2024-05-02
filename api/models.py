from django.db import models

# Create your models here.
class KerasModel(models.Model):
    
    name = models.CharField(verbose_name="Tên mô hình", max_length=255)
    desc = models.TextField(verbose_name="Mô tả", null=True, blank=True)
    save_file = models.FileField(verbose_name="Save File")
    
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateField(auto_now=True)
    
    