from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import requests
import json
from django.conf import settings
import base64
import time

# Global variable to cache Nyckel access token
nyckel_access_token = {
    "token": None,
    "expiry_time": 0
}

# Load trained model
model = tf.keras.models.load_model("plant_growth_stage.h5")
CLASS_NAMES = ["Seedling", "Vegetative_Early", "Vegetative_Late", "Flowering", "Fruiting_Ripe","Fruiting_Unripe"]

def _get_nyckel_access_token():
    global nyckel_access_token
    # Check if token is still valid
    if nyckel_access_token["token"] and nyckel_access_token["expiry_time"] > time.time():
        return nyckel_access_token["token"]

    # If not, get a new token
    try:
        token_url = settings.NYCKEL_TOKEN_URL
        client_id = settings.NYCKEL_CLIENT_ID
        client_secret = settings.NYCKEL_CLIENT_SECRET

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }

        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        token_data = response.json()

        nyckel_access_token["token"] = token_data["access_token"]
        # Set expiry time a bit before actual expiry to be safe
        nyckel_access_token["expiry_time"] = time.time() + token_data["expires_in"] - 60
        return nyckel_access_token["token"]

    except requests.exceptions.RequestException as e:
        print(f"Error getting Nyckel access token: {e}")
        return None

def _classify_image_with_nyckel(image_bytes, filename="unknown.jpg"):
    token = _get_nyckel_access_token()
    if not token:
        return None, None

    try:
        # Infer MIME type from filename
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.jpg' or ext == '.jpeg':
            mime_type = 'image/jpeg'
        elif ext == '.png':
            mime_type = 'image/png'
        else:
            # Fallback for other types, or raise an error
            mime_type = 'application/octet-stream'

        invoke_url = settings.NYCKEL_INVOKE_URL
        headers = {
            'Authorization': 'Bearer ' + token,
            'Content-Type': 'application/json'
        }

        # Construct the full data URI
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        data_uri = f"data:{mime_type};base64,{base64_image}"
        data = {"data": data_uri}

        response = requests.post(invoke_url, headers=headers, json=data)
        response.raise_for_status()
        nyckel_results = response.json()
        
        # Nyckel API returns a dictionary directly
        if nyckel_results and isinstance(nyckel_results, dict) and nyckel_results.get('labelName'):
            stage = nyckel_results['labelName']
            confidence = nyckel_results.get('confidence') # Confidence might be optional depending on API
            if confidence is not None:
                return stage, float(confidence)
            else:
                return stage, None
        else:
            print(f"Nyckel API response did not contain expected prediction format: {nyckel_results}")
            return None, None

    except requests.exceptions.RequestException as e:
        print(f"Error classifying image with Nyckel API: {e}")
        return None, None

def index(request):
    prediction = None # Keep prediction for now, can be removed later if not used
    confidence = None
    img_url = None
    stage = None # Ensure stage is initialized for template context
    error_message = None # Initialize error message

    if request.method == "POST" and request.FILES.get("plant_image"):
        uploaded_file = request.FILES["plant_image"]
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        img_url = fs.url(filename)

        # Read image bytes for Nyckel API (if used)
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        image_bytes = uploaded_file.read()

        if request.POST.get("use_nyckel") == "on":
            stage, confidence = _classify_image_with_nyckel(image_bytes, filename)
            if stage and confidence is not None:
                pass # stage and confidence are already set
            else:
                error_message = "Nyckel API failed to classify the image. Check console for details."
        else:
            try:
                # Preprocess for local model
                img_path = os.path.join(fs.location, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction with local model
                preds = model.predict(img_array)
                stage = CLASS_NAMES[np.argmax(preds)]
                confidence = float(np.max(preds))
            except Exception as e:
                error_message = f"Local model classification failed: {e}"
                stage = None
                confidence = None

    return render(request, "index.html", {
        "img_url": img_url,
        "stage": stage,
        "confidence": confidence,
        "error_message": error_message
    })