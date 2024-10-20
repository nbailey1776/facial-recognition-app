import cv2
import os
import random
import numpy as np

class DataCollector:
    def __init__(self, user_id, name, dataset_dir='static/datasets', file_uploads=None):
        self.user_id = user_id
        self.name = name
        self.dataset_dir = dataset_dir
        self.video = None  # Only used if we are using the webcam
        self.file_uploads = file_uploads  # List of uploaded image paths

        # Update the path to the Haar Cascade XML file
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.facedetect = cv2.CascadeClassifier(cascade_path)

        self.count = 0
        self.max_images = 500

        # Set dataset path for the user
        self.dataset_path = os.path.join(self.dataset_dir, f'{self.name}_{self.user_id}')
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def start_collection(self):
        """Start the dataset collection either from a webcam or uploaded image files."""
        # Ensure the dataset directory exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        if self.file_uploads:
            # Process each uploaded image
            for file_path in self.file_uploads:
                self.process_uploaded_image(file_path)
            self.duplicate_images()
        else:
            # Start collecting data from webcam
            self.video = cv2.VideoCapture(0)
            self.collect_from_webcam()

    def collect_from_webcam(self):
        """Collect face data from webcam feed."""
        print(f"\n[INFO] Starting dataset collection for {self.name} (ID: {self.user_id}). Look at the camera...")
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            self.process_frame(frame)

            # Display live video
            cv2.imshow("Frame", frame)

            # Stop collecting after max images
            if self.count >= self.max_images:
                break

            # Allow early quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_collection()

    def process_uploaded_image(self, image_path):
        """Process each uploaded image file."""
        print(f"\n[INFO] Processing uploaded image {image_path} for {self.name} (ID: {self.user_id})...")

        if not os.path.isfile(image_path):
            print(f"[ERROR] File {image_path} does not exist.")
            return

        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Unable to read the uploaded image: {image_path}")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"[WARNING] No faces detected in {image_path}. Skipping this image.")
            return

        for (x, y, w, h) in faces:
            self.count += 1
            face = gray[y:y + h, x:x + w]
            # Save the face image to the dataset
            face_filename = f'User_{self.name}_{self.user_id}_{self.count}.jpg'
            cv2.imwrite(os.path.join(self.dataset_path, face_filename), face)

        print(f"\n[INFO] Image {image_path} processed successfully.")

    def duplicate_images(self):
        """Duplicate existing images until the total count reaches self.max_images."""
        existing_images = [f for f in os.listdir(self.dataset_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        num_existing = len(existing_images)

        if num_existing == 0:
            print("[ERROR] No images to duplicate.")
            return

        print(f"\n[INFO] Duplicating images to reach {self.max_images} images.")

        while self.count < self.max_images:
            for img_name in existing_images:
                if self.count >= self.max_images:
                    break
                img_path = os.path.join(self.dataset_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                self.count += 1
                new_filename = f'User_{self.name}_{self.user_id}_{self.count}.jpg'
                cv2.imwrite(os.path.join(self.dataset_path, new_filename), image)

        print(f"[INFO] Image duplication complete. Total images: {self.count}")


    def process_frame(self, frame):
        """Process each frame from the webcam."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            self.count += 1
            face = gray[y:y + h, x:x + w]
            # Save the face image to the dataset
            face_filename = f'User_{self.name}_{self.user_id}_{self.count}.jpg'
            cv2.imwrite(os.path.join(self.dataset_path, face_filename), face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.putText(frame, f'Captured: {self.count}/{self.max_images}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def stop_collection(self):
        if self.video:
            self.video.release()
        cv2.destroyAllWindows()
        print(f"\n[INFO] Dataset collection for {self.name} completed.")
