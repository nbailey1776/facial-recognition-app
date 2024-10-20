import cv2

class FaceRecognizer:
    def __init__(self, model_path='Trainer.yml', name_dict=None):
        self.video = cv2.VideoCapture(0)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.facedetect = cv2.CascadeClassifier(cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)
        self.name_dict = name_dict if name_dict is not None else {0: "Unknown"}

    def recognize(self):
        print("[INFO] Starting real-time face recognition...")
        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                user_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                if confidence < 80:  # Adjust this threshold as necessary
                    name = self.name_dict.get(user_id, "Unknown")  # Safely map user_id to name
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()
        print("[INFO] Real-time face recognition ended.")
