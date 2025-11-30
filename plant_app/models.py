from django.db import models
from django.contrib.auth.models import User

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='predictions/')
    predicted_stage = models.CharField(max_length=100)
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_correct = models.BooleanField(null=True, blank=True) # For manual validation later

    def __str__(self):
        return f"{self.user.username} - {self.predicted_stage} ({self.created_at})"
