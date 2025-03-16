from django.urls import path
from .views import recommend_material

urlpatterns = [
    path("recommend/", recommend_material, name="recommend_material"),
]
