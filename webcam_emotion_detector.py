import cv2
from fer import FER

cap = cv2.VideoCapture(0)
#capturing video input. If there is only one web cam, then 0. If you want to access the 2nd webcam, 
# then give the number as 1, 3rd webcam then the number as 2 and so on.

detector = FER(mtcnn=True)
#FER is a class from fer library - Facial Expression Recognizer
#mtcnn stands for: Multi-task Cascaded Convolutional Networks — a specialized face detection model.
#this mtcnn=True means - "Detect faces before recognizing emotions"
#and mtcnn=True because it is more accurate than the default face detector and detects multiple faces

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    #Captures one frame from your webcam
    #ret is True if frame was captured successfully
    #frame is the image data (a NumPy array of pixels)
    if not ret:
        break
    #If the camera failed, it exits the loop.

    # Detect emotions in the current frame
    results = detector.detect_emotions(frame)
    #The FER detector scans the frame:
    #Detects faces
    #Predicts emotion scores for each face

    # Loop through detected faces
    for result in results:
        (x, y, w, h) = result["box"]
        emotions = result["emotions"]
        top_emotion = max(emotions, key=emotions.get)

        # Draw rectangle and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, top_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Emotion Detection', frame)

    # Quit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()