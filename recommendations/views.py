from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from .ml_model import predict_price

def recommend_material(request):
    if request.method == "POST":
        cement_quality = request.POST["cement_quality"]
        # cement_price = float(request.POST["cement_price"])
        brick_quality = request.POST["brick_quality"]
        # brick_price = float(request.POST["brick_price"])
        sand_quality = request.POST["sand_quality"]
        # sand_price = float(request.POST["sand_price"])
        iron_quality = request.POST["iron_quality"]
        # iron_price = float(request.POST["iron_price"])
        env_condition = request.POST["env_condition"]

        predicted_price = predict_price(cement_quality, brick_quality,
                                        sand_quality,  iron_quality,  env_condition)

        return render(request, "recommendations/results.html", {"predicted_price": predicted_price})

    return render(request, "recommendations/form.html")
