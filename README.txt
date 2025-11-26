Plant Growth Stage Classification

This branch (`v2`) contains the code for training the model and the initial version of the Django application.

**Note on Model Files:**
The trained model files (`.h5`) are too large to be stored directly in this branch's history. 
To run the application, please download the `plant_growth_stage.h5` file from the `v3` branch (where it is stored using Git LFS) or train the model yourself using `train_model.py`.

**Setup:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run training (optional): `python train_model.py`
3. Start the app: `python manage.py runserver`
