from fer import FER
import matplotlib.pyplot as plt # to display image
import cv2 # for image loading and drawing

# Loading the image
image_path = 'amogh.jpeg'
# did not work for angry:
# {'angry': 0.32, 'disgust': 0.0, 'fear': 0.16, 'happy': 0.48, 'sad': 0.01, 'surprise': 0.02, 'neutral': 0.0}

#happy photo:
# {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 1.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}

#my photo
#{'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.19, 'sad': 0.01, 'surprise': 0.0, 'neutral': 0.79}
image = cv2.imread(image_path)

#Initialize FER detector
detector = FER(mtcnn = True) # mtcnn = True improves face detection but requires a bit more computation

results = detector.detect_emotions(image) # returns a list of faces with their emotion scores and bounding boxes.

print("Emotion detection results")
for face in results:
    print(face["emotions"])

for face in results:
    (x, y, w, h) = face["box"]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    top_emotion = max(face["emotions"], key = face["emotions"].get)
    cv2.putText(image, top_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# Draws bounding boxes around detected faces and labels the dominant emotion.

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()


