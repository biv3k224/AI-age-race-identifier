import urllib.request
import os
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Create model directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Race model files
race_files = {
    'race_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/race_net.caffemodel',
    'race_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/race_deploy.prototxt'
}

print("Downloading race model files...")
for filename, url in race_files.items():
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, f'models/{filename}')
        print(f"✓ {filename} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")

print("Download process completed!")