import os
import requests

def load_model(model_name, model_cache_dir='/models'):
    local_model_path = os.path.join(model_cache_dir, model_name)
    if os.path.exists(local_model_path):
        return local_model_path  # Load from local cache
    
    # If model not found locally, download it
    url = f"https://example.com/models/{model_name}"
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_model_path, 'wb') as f:
            f.write(response.content)
        return local_model_path
    else:
        raise FileNotFoundError(f"Model {model_name} not found.")