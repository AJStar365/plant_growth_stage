from django.shortcuts import render, redirect, get_object_or_404
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
from .models import Prediction
from django.contrib.auth.decorators import login_required, user_passes_test
from django.db.models import Count
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

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
    all_predictions = None # To store detailed breakdown

    if request.method == "POST" and request.FILES.get("plant_image"):
        uploaded_file = request.FILES["plant_image"]
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        img_url = fs.url(filename)

        # Read image bytes for Nyckel API
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        image_bytes = uploaded_file.read()

        # 1. Try Nyckel API first
        stage, confidence = _classify_image_with_nyckel(image_bytes, filename)
        
        # If Nyckel succeeds, we currently don't get full breakdown, so all_predictions stays None
        # Or we could wrap the single result into a list if we wanted consistent structure
        if stage:
             all_predictions = [{'label': stage, 'score': confidence}]

        # 2. If Nyckel fails (stage is None), fallback to local model
        if stage is None:
            print("Nyckel API failed, falling back to local model...")
            try:
                # Preprocess for local model
                img_path = os.path.join(fs.location, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction with local model
                preds = model.predict(img_array)[0] # Get probabilities for the single image
                
                # Create detailed breakdown
                all_predictions = []
                for i, score in enumerate(preds):
                    all_predictions.append({'label': CLASS_NAMES[i], 'score': float(score)})
                
                # Sort by score descending
                all_predictions.sort(key=lambda x: x['score'], reverse=True)

                stage = all_predictions[0]['label']
                confidence = all_predictions[0]['score']
                
                # Clear error message if local model succeeds
                error_message = None 
            except Exception as e:
                error_message = f"Both Nyckel API and Local model classification failed. Local Error: {e}"
                stage = None
                confidence = None
                all_predictions = None

        # Save prediction if user is logged in and prediction was successful
        if request.user.is_authenticated and stage:
            # filename returned by fs.save is relative to MEDIA_ROOT
            Prediction.objects.create(
                user=request.user,
                image=filename, 
                predicted_stage=stage,
                confidence=confidence
            )

    return render(request, "index.html", {
        "img_url": img_url,
        "stage": stage,
        "confidence": confidence,
        "error_message": error_message,
        "all_predictions": all_predictions
    })

@login_required
def dashboard(request):
    # Get user's predictions
    user_predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')

    # Aggregate data for chart (e.g., count of each stage)
    stage_counts = user_predictions.values('predicted_stage').annotate(count=Count('predicted_stage'))
    
    # Prepare data for Chart.js
    chart_labels = [item['predicted_stage'] for item in stage_counts]
    chart_data = [item['count'] for item in stage_counts]

    return render(request, "dashboard.html", {
        "predictions": user_predictions,
        "chart_labels": json.dumps(chart_labels),
        "chart_data": json.dumps(chart_data)
    })

def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/signup.html', {'form': form})

@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard(request):
    users = User.objects.all()
    return render(request, 'admin_dashboard.html', {'users': users})

@user_passes_test(lambda u: u.is_superuser)
def admin_user_dashboard(request, user_id):
    target_user = get_object_or_404(User, pk=user_id)
    
    # Get target user's predictions
    user_predictions = Prediction.objects.filter(user=target_user).order_by('-created_at')

    # Aggregate data for chart
    stage_counts = user_predictions.values('predicted_stage').annotate(count=Count('predicted_stage'))
    
    # Prepare data for Chart.js
    chart_labels = [item['predicted_stage'] for item in stage_counts]
    chart_data = [item['count'] for item in stage_counts]

    return render(request, "dashboard.html", {
        "predictions": user_predictions,
        "chart_labels": json.dumps(chart_labels),
        "chart_data": json.dumps(chart_data),
        "viewing_user": target_user # To show who we are viewing
    })