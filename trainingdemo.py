import os
import cv2
import numpy as np

class FaceTrainer:
    def __init__(self, dataset_dir='static/datasets', model_path='Trainer.yml'):
        self.dataset_dir = dataset_dir
        self.model_path = model_path

    def train(self):
        faces = []
        ids = []

        image_paths = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("png"):
                    image_paths.append(os.path.join(root, file))

        for image_path in image_paths:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            user_id = int(os.path.basename(os.path.dirname(image_path)).split('_')[-1])
            faces.append(np.array(gray, 'uint8'))
            ids.append(user_id)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(self.model_path)
        print("[INFO] Training completed and model saved.")
