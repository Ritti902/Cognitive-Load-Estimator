  # save as test_mediapipe.py
import cv2
import mediapipe as mp
  
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
  
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
  
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)
  
if results.multi_face_landmarks:
    print('Face detected!')
    landmarks = results.multi_face_landmarks[0].landmark
      # Print position of landmark 33 (left eye outer corner)
    lm = landmarks[33]
    print(f'Landmark 33: x={lm.x:.3f}, y={lm.y:.3f}')
else:
    print('No face detected. Make sure your face is visible.')
  
cap.release()
  
