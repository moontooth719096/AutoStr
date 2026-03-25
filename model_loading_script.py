import os
import urllib.request

MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/models')

def load_model(model_name):
    model_path = os.path.join(MODEL_CACHE_DIR, model_name)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return model_path  # Return the path to the cached model
    else:
        print(f"Model not found locally. Downloading from remote...")
        # Code to download the model
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        remote_url = f'http://example.com/models/{model_name}'  # Example URL
        urllib.request.urlretrieve(remote_url, model_path)
        return model_path  # Return the path to the downloaded model
