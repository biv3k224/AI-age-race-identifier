import cv2
import numpy as np

def load_age_model():
    """Load the age estimation model"""
    try:
        age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
        print("✓ Age model loaded successfully")
        return age_net
    except Exception as e:
        print(f"✗ Error loading age model: {e}")
        return None

def load_face_detector():
    """Load face detection model"""
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✓ Face detector loaded successfully")
        return face_detector
    except Exception as e:
        print(f"✗ Error loading face detector: {e}")
        return None

def detect_faces(face_detector, frame):
    """Optimized face detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),  # Increased for better accuracy
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
    if value > 180 and saturation < 100:  # High brightness, low saturation
        scores['Caucasian'] += 2
    if lightness > 160:  # Very high lightness in LAB
        scores['Caucasian'] += 2
    
    # East Asian features (light to medium yellowish skin)
    if 140 <= red <= 190 and 120 <= green <= 170 and 100 <= blue <= 150:
        scores['East Asian'] += 3
    if 15 <= hue <= 30:  # Yellowish hues
        scores['East Asian'] += 2
    if cb_channel > 125:  # Higher Cb values typical in East Asians
        scores['East Asian'] += 2
    if 130 <= lightness <= 170:  # Medium-high lightness
        scores['East Asian'] += 1
    
    # South Asian features (medium olive to brown skin)
    if 100 <= red <= 160 and 80 <= green <= 140 and 60 <= blue <= 120:
        scores['South Asian'] += 4
    if 10 <= hue <= 25:  # Yellowish-reddish hues
        scores['South Asian'] += 2
    if 120 <= cr_channel <= 150:  # Specific Cr range
        scores['South Asian'] += 2
    if 100 <= lightness <= 140:  # Medium lightness
        scores['South Asian'] += 1
    
    # African features (dark brown to black skin)
    if red < 110 and green < 100 and blue < 90:
        scores['African'] += 4
    if value < 130:  # Low brightness
        scores['African'] += 2
    if lightness < 110:  # Low lightness
        scores['African'] += 2
    if saturation > 80:  # Can have good saturation despite dark skin
        scores['African'] += 1
    
    # Middle Eastern features (olive to tan skin)
    if 120 <= red <= 170 and 100 <= green <= 150 and 80 <= blue <= 130:
        scores['Middle Eastern'] += 3
    if 8 <= hue <= 22:  # Olive tones
        scores['Middle Eastern'] += 2
    if 110 <= lightness <= 150:  # Medium lightness
        scores['Middle Eastern'] += 1
    if 110 <= cr_channel <= 140:  # Cr range
        scores['Middle Eastern'] += 1
    
    # Latino/Hispanic features (varied from light to medium-brown)
    if 130 <= red <= 180 and 110 <= green <= 160 and 90 <= blue <= 140:
        scores['Latino/Hispanic'] += 3
    if 12 <= hue <= 28:  # Warm tones
        scores['Latino/Hispanic'] += 2
    if 120 <= lightness <= 160:  # Medium lightness range
        scores['Latino/Hispanic'] += 1
    
    # Find best ethnicity
    max_score = max(scores.values())
    best_ethnicities = [eth for eth, score in scores.items() if score == max_score]
    
    # If tie, use the first one (or add tie-breaking rules)
    ethnicity = best_ethnicities[0]
    
    # Calculate confidence based on score dominance
    sorted_scores = sorted(scores.values(), reverse=True)
    if max_score == 0:
        confidence = 0.5  # Low confidence if no strong features
    elif len(sorted_scores) > 1:
        score_diff = sorted_scores[0] - sorted_scores[1]
        confidence = min(0.95, 0.6 + (score_diff * 0.1))
    else:
        confidence = 0.7
    
    return ethnicity, confidence, avg_bgr

def main():
    # Load models
    print("Loading models...")
    age_net = load_age_model()
    face_detector = load_face_detector()
    
    if age_net is None or face_detector is None:
        print("Failed to load models. Exiting.")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("=== Age & Ethnicity Estimation ===")
    print("Press 'q' to quit")
    print("Make sure face is well-lit for best results")
    print("===================================")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        faces = detect_faces(face_detector, frame)
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Estimate age and ethnicity
            try:
                age, age_confidence = estimate_age(age_net, face_roi)
                ethnicity, eth_confidence, avg_bgr = estimate_ethnicity(face_roi)
                
                # Display information with better formatting
                cv2.putText(frame, f'Person {i+1}', (x, y-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f'Age: {age}', (x, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'Ethnicity: {ethnicity}', (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f'Age Conf: {age_confidence:.2f}', (x, y+h+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(frame, f'Eth Conf: {eth_confidence:.2f}', (x, y+h+55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
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
        cv2.imshow('Age & Ethnicity Estimation', frame)
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Estimation stopped. Thank you for using the app!")

if __name__ == "__main__":
    main()