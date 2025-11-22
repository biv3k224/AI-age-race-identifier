import urllib.request
import os
import ssl

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Create model directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Updated model files with working URLs
model_files = {
    'age_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel',
    'age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt',
    'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
    'opencv_face_detector_uint8.pb': 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/opencv_face_detector_uint8.pb'
}

print("Downloading model files with corrected URLs...")
for filename, url in model_files.items():
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, f'models/{filename}')
        print(f"✓ {filename} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        
print("Download process completed!")