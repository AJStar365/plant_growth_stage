from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load trained model
model = tf.keras.models.load_model("plant_growth_stage.h5")
CLASS_NAMES = ["Seedling", "Vegetative_Early", "Vegetative_Late", "Flowering", "Fruiting_Ripe","Fruiting_Unripe"]

def index(request):
    prediction = None
    confidence = None
    img_url = None

    if request.method == "POST" and request.FILES.get("plant_image"):
        uploaded_file = request.FILES["plant_image"]
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        img_url = fs.url(filename)

        # Preprocess
        img_path = os.path.join(fs.location, filename)
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        stage = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        prediction = f"{stage} ({confidence:.2f})"

    return render(request, "index.html", {
        "prediction": prediction,
        "img_url": img_url
    })
