# import os
# import cv2
# from deepface import DeepFace

# # Paths to your folders
# KNOWN_FOLDER = "Face Recog/Known Faces"
# UNKNOWN_FOLDER = "Face Recog/Unknown Faces"

# # Loop through unknown images
# for unknown_file in os.listdir(UNKNOWN_FOLDER):
#     unknown_path = os.path.join(UNKNOWN_FOLDER, unknown_file)
#     print(f"\n🔍 Checking {unknown_file}")

#     found_match = False

#     # Compare with each known image
#     for known_file in os.listdir(KNOWN_FOLDER):
#         known_path = os.path.join(KNOWN_FOLDER, known_file)

#         try:
#             result = DeepFace.verify(img1_path=unknown_path, img2_path=known_path, enforce_detection=False)

#             if result["verified"]:
#                 print(f"✅ Match found! {unknown_file} matches {known_file}")
#                 found_match = True
#                 break  # No need to check more if match is found

#         except Exception as e:
#             print(f"Error comparing {unknown_file} and {known_file}: {e}")

#     if not found_match:
#         print(f"❌ No match found for {unknown_file}")

import os
import cv2
from deepface import DeepFace

# Paths to your folders
KNOWN_FOLDER = "Face Recog/Known Faces"
UNKNOWN_FOLDER = "Face Recog/Unknown Faces"

# Loop through unknown images
for unknown_file in os.listdir(UNKNOWN_FOLDER):
    unknown_path = os.path.join(UNKNOWN_FOLDER, unknown_file)
    print(f"\n🔍 Checking {unknown_file}")

    found_match = False

    # Read the unknown image
    unknown_img = cv2.imread(unknown_path)

    try:
        # Extract faces from the unknown image using DeepFace
        faces = DeepFace.extract_faces(unknown_path, enforce_detection=False)

        if faces:
            # Iterate over detected faces
            for face in faces:
                # Ensure 'region' exists in the face data
                if "region" in face:
                    region = face["region"]
                    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                    # Draw bounding box around the face
                    cv2.rectangle(unknown_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

            # Show image with bounding box
            cv2.imshow(f"Face Detection - {unknown_file}", unknown_img)
            cv2.waitKey(0)

        else:
            print(f"No faces detected in {unknown_file}")
            
        # Compare with each known image
        for known_file in os.listdir(KNOWN_FOLDER):
            known_path = os.path.join(KNOWN_FOLDER, known_file)

            try:
                result = DeepFace.verify(img1_path=unknown_path, img2_path=known_path, enforce_detection=False)

                if result["verified"]:
                    print(f"✅ Match found! {unknown_file} matches {known_file}")
                    found_match = True
                    break  # No need to check more if match is found

            except Exception as e:
                print(f"Error comparing {unknown_file} and {known_file}: {e}")

        if not found_match:
            print(f" No match found for {unknown_file}")

    except Exception as e:
        print(f"Error extracting faces in {unknown_file}: {e}")

cv2.destroyAllWindows()
