import cv2
import numpy as np

def load_models():
    """Load all models: age, gender, and face detector"""
    models = {}
    
    # Load age model
    try:
        models['age_net'] = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
        print("✓ Age model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading age model: {e}")
        models['age_net'] = None
    
    # Load gender model
    try:
        models['gender_net'] = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')
        print("✓ Gender model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading gender model: {e}")
        # If model file doesn't exist, we'll use a fallback method
        models['gender_net'] = None
    
    # Load face detector
    try:
        models['face_detector'] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✓ Face detector loaded successfully")
    except Exception as e:
        print(f"✗ Error loading face detector: {e}")
        models['face_detector'] = None
    
    return models

def detect_faces(face_detector, frame):
    """Face detection with balanced parameters"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def estimate_age(age_net, face_roi):
    """Estimate age with custom ranges"""
    age_list = [
        '(0-5)',    # Baby/Toddler
        '(6-12)',   # Child
        '(13-19)',  # Teenager
        '(20-30)',  # Young Adult
        '(31-45)',  # Adult
        '(46-60)',  # Middle-aged
        '(61+)'     # Senior
    ]
    
    # Preprocess face for age estimation
    blob = cv2.dnn.blobFromImage(
        face_roi, 
        1.0, 
        (227, 227), 
        (78.4263377603, 87.7689143744, 114.895847746), 
        swapRB=False
    )
    
    # Get age prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax() % len(age_list)]
    
    # Get confidence score
    confidence = age_preds[0].max()
    
    return age, confidence

def estimate_gender(gender_net, face_roi):
    """Estimate gender using the gender model"""
    gender_list = ['Male', 'Female']
    
    if gender_net is None:
        # Fallback: simple rule-based estimation if model not available
        return estimate_gender_fallback(face_roi)
    
    # Preprocess face for gender estimation
    blob = cv2.dnn.blobFromImage(
        face_roi, 
        1.0, 
        (227, 227), 
        (78.4263377603, 87.7689143744, 114.895847746), 
        swapRB=False
    )
    
    # Get gender prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    
    # Get confidence score
    confidence = gender_preds[0].max()
    
    return gender, confidence

def estimate_gender_fallback(face_roi):
    """
    Fallback gender estimation based on facial features
    This is less accurate but works when the model is not available
    """
    # Analyze face shape and features
    height, width = face_roi.shape[:2]
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Simple feature analysis (jawline width vs height)
    aspect_ratio = width / height
    
    # Typically male faces are more square, female more oval
    if aspect_ratio > 0.85:  # More square
        gender = 'Male'
        confidence = 0.65
    else:  # More oval
        gender = 'Female'
        confidence = 0.65
    
    return gender, confidence

def estimate_ethnicity(face_roi):
    """
    Balanced ethnicity estimation for all major ethnic groups
    """
    ethnicity_list = ['East Asian', 'South Asian', 'Caucasian', 'African', 'Middle Eastern', 'Latino/Hispanic']
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    ycrbr = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
    
    # Calculate color statistics
    avg_bgr = np.mean(face_roi, axis=(0, 1))
    avg_hsv = np.mean(hsv, axis=(0, 1))
    avg_lab = np.mean(lab, axis=(0, 1))
    avg_ycrbr = np.mean(ycrbr, axis=(0, 1))
    
    blue, green, red = avg_bgr
    hue, saturation, value = avg_hsv
    lightness, a_channel, b_channel = avg_lab
    y_channel, cr_channel, cb_channel = avg_ycrbr
    
    # Reset scores for all ethnicities
    scores = {eth: 0 for eth in ethnicity_list}
    
    # Caucasian features (very light skin)
    if red > 180 and green > 170 and blue > 160:
        scores['Caucasian'] += 4
    if value > 180 and saturation < 100:
        scores['Caucasian'] += 2
    if lightness > 160:
        scores['Caucasian'] += 2
    
    # East Asian features (light to medium yellowish skin)
    if 140 <= red <= 190 and 120 <= green <= 170 and 100 <= blue <= 150:
        scores['East Asian'] += 3
    if 15 <= hue <= 30:
        scores['East Asian'] += 2
    if cb_channel > 125:
        scores['East Asian'] += 2
    if 130 <= lightness <= 170:
        scores['East Asian'] += 1
    
    # South Asian features (medium olive to brown skin)
    if 100 <= red <= 160 and 80 <= green <= 140 and 60 <= blue <= 120:
        scores['South Asian'] += 4
    if 10 <= hue <= 25:
        scores['South Asian'] += 2
    if 120 <= cr_channel <= 150:
        scores['South Asian'] += 2
    if 100 <= lightness <= 140:
        scores['South Asian'] += 1
    
    # African features (dark brown to black skin)
    if red < 110 and green < 100 and blue < 90:
        scores['African'] += 4
    if value < 130:
        scores['African'] += 2
    if lightness < 110:
        scores['African'] += 2
    if saturation > 80:
        scores['African'] += 1
    
    # Middle Eastern features (olive to tan skin)
    if 120 <= red <= 170 and 100 <= green <= 150 and 80 <= blue <= 130:
        scores['Middle Eastern'] += 3
    if 8 <= hue <= 22:
        scores['Middle Eastern'] += 2
    if 110 <= lightness <= 150:
        scores['Middle Eastern'] += 1
    if 110 <= cr_channel <= 140:
        scores['Middle Eastern'] += 1
    
    # Latino/Hispanic features (varied from light to medium-brown)
    if 130 <= red <= 180 and 110 <= green <= 160 and 90 <= blue <= 140:
        scores['Latino/Hispanic'] += 3
    if 12 <= hue <= 28:
        scores['Latino/Hispanic'] += 2
    if 120 <= lightness <= 160:
        scores['Latino/Hispanic'] += 1
    
    # Find best ethnicity
    max_score = max(scores.values())
    best_ethnicities = [eth for eth, score in scores.items() if score == max_score]
    ethnicity = best_ethnicities[0]
    
    # Calculate confidence
    sorted_scores = sorted(scores.values(), reverse=True)
    if max_score == 0:
        confidence = 0.5
    elif len(sorted_scores) > 1:
        score_diff = sorted_scores[0] - sorted_scores[1]
        confidence = min(0.95, 0.6 + (score_diff * 0.1))
    else:
        confidence = 0.7
    
    return ethnicity, confidence, avg_bgr

def main():
    # Load all models
    print("Loading all models...")
    models = load_models()
    
    if models['face_detector'] is None:
        print("Failed to load face detector. Exiting.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("=== Complete Demographic Analysis ===")
    print("Features: Age, Gender, and Ethnicity Estimation")
    print("Press 'q' to quit")
    print("Make sure face is well-lit for best results")
    print("=====================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = detect_faces(models['face_detector'], frame)
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face with different colors based on position
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Estimate demographics
            try:
                # Age estimation
                age, age_confidence = "Unknown", 0.0
                if models['age_net'] is not None:
                    age, age_confidence = estimate_age(models['age_net'], face_roi)
                
                # Gender estimation
                gender, gender_confidence = "Unknown", 0.0
                gender, gender_confidence = estimate_gender(models['gender_net'], face_roi)
                
                # Ethnicity estimation
                ethnicity, eth_confidence, avg_bgr = estimate_ethnicity(face_roi)
                
                # Display information with organized layout
                cv2.putText(frame, f'Person {i+1}', (x, y-55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Age (Green)
                cv2.putText(frame, f'Age: {age}', (x, y-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'({age_confidence:.2f})', (x+80, y-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Gender (Blue)
                gender_color = (255, 0, 0)  # Blue for male, Magenta for female
                if gender == 'Female':
                    gender_color = (255, 0, 255)  # Magenta
                
                cv2.putText(frame, f'Gender: {gender}', (x, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, gender_color, 2)
                cv2.putText(frame, f'({gender_confidence:.2f})', (x+100, y-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, gender_color, 1)
                
                # Ethnicity (Orange)
                cv2.putText(frame, f'Ethnicity: {ethnicity}', (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.putText(frame, f'({eth_confidence:.2f})', (x+120, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
            except Exception as e:
                print(f"Estimation error: {e}")
        
        # Display frame info
        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f'Faces detected: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Complete Demographic Analysis', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Analysis stopped. Thank you for using the app!")

if __name__ == "__main__":
    main()