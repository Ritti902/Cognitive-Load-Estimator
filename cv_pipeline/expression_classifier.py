import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
  
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
  
def extract_expression_features(landmarks):
      '''
      Extracts 6 numbers from face landmarks.
      These 6 numbers describe the face's expression.
      '''
      def dist(a, b):
          return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)
  
      # Feature 1: Left eyebrow raise
      # Landmark 70 = left eyebrow, 159 = left eye top
      f1 = dist(landmarks[70], landmarks[159])
  
      # Feature 2: Right eyebrow raise
      f2 = dist(landmarks[300], landmarks[386])
  
      # Feature 3: Lip press (how close upper and lower lip are)
      # 13 = upper lip, 14 = lower lip
      f3 = dist(landmarks[13], landmarks[14])
  
      # Feature 4: Jaw open (chin vs nose distance)
      # 152 = chin, 1 = nose tip
      f4 = dist(landmarks[152], landmarks[1])
  
      # Feature 5: Left eye openness (EAR-like)
      f5 = dist(landmarks[159], landmarks[145])
  
      # Feature 6: Forehead tension (brow furrow)
      f6 = dist(landmarks[9], landmarks[107])
  
      return [f1, f2, f3, f4, f5, f6]
  
  
def train_expression_classifier(X, y):
      '''
      X = list of feature arrays
      y = list of labels: 0=NEUTRAL, 1=CONFUSED, 2=FOCUSED
      Call this once during Month 3 data collection.
      '''
      clf = RandomForestClassifier(n_estimators=100, random_state=42)
      clf.fit(X, y)
      with open('expression_model.pkl', 'wb') as f:
          pickle.dump(clf, f)
      print('Expression model saved!')
      return clf
  
  
def load_or_default_classifier():
      '''Load saved model, or return None if not trained yet'''
      if os.path.exists('expression_model.pkl'):
          with open('expression_model.pkl', 'rb') as f:
              return pickle.load(f)
      return None   # will use rule-based fallback
  
  
def predict_expression(features, clf=None):
      '''Predict expression from 6 features'''
      if clf is not None:
          pred = clf.predict([features])[0]
          return ['NEUTRAL', 'CONFUSED', 'FOCUSED'][pred]
  
      # Rule-based fallback (before model is trained)
      brow_raise = (features[0] + features[1]) / 2
      lip_press  = features[2]
      if brow_raise > 0.04:
          return 'CONFUSED'
      elif lip_press < 0.02:
          return 'FOCUSED'
      else:
          return 'NEUTRAL'
