from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datacollect import DataCollector
from trainingdemo import FaceTrainer
from testmodel import FaceRecognizer
import os
import shutil

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///default.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
db = SQLAlchemy(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database model for users
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, unique=True, nullable=False)

# Function to load user data and return a dictionary mapping user_id to name
def load_names():
    users = User.query.all()
    name_dict = {user.user_id: user.name for user in users}
    return name_dict

# Helper function to get user image
@app.context_processor
def utility_processor():
    def get_user_image(name, user_id):
        dataset_folder = os.path.join('datasets', f'{name}_{user_id}')
        dataset_path = os.path.join(app.root_path, 'static', dataset_folder)

        # Ensure the directory exists and has images
        if os.path.exists(dataset_path):
            images = [file for file in os.listdir(dataset_path) if file.lower().endswith(('jpg', 'jpeg', 'png'))]
            if images:
                # Return the relative path to the image for use in the template
                return os.path.join(dataset_folder, images[0]).replace("\\", "/")
        return 'default.jpg'  # Return a default image if no images are found
    return dict(get_user_image=get_user_image)



# Initialize database
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/collect', methods=['GET', 'POST'])
def collect_data():
    if request.method == 'POST':
        user_id = request.form['user_id']
        name = request.form['name']
        uploaded_files = request.files.getlist('files')  # Handle multiple files

        # Filter out empty files
        uploaded_files = [file for file in uploaded_files if file.filename != '']

        # Convert user_id to integer and validate
        try:
            user_id = int(user_id)
            if user_id <= 0:
                raise ValueError
        except ValueError:
            flash("User ID must be a positive integer.", "danger")
            return redirect(url_for('collect_data'))

        # Check for duplicate user_id and name in the database
        existing_user_by_id = User.query.filter_by(user_id=user_id).first()
        existing_user_by_name = User.query.filter_by(name=name).first()

        if existing_user_by_id:
            flash(f"User ID {user_id} is already taken. Please choose a different ID.", "danger")
            return redirect(url_for('collect_data'))
        if existing_user_by_name:
            flash(f"Name '{name}' is already taken. Please choose a different name.", "danger")
            return redirect(url_for('collect_data'))

        # Save the new user to the database
        new_user = User(user_id=user_id, name=name)
        db.session.add(new_user)
        db.session.commit()

        # Proceed with data collection
        if uploaded_files:
            file_paths = []
            upload_folder = app.config['UPLOAD_FOLDER']

            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            for uploaded_file in uploaded_files:
                # Save each uploaded file to the UPLOAD_FOLDER
                file_path = os.path.join(upload_folder, uploaded_file.filename)
                uploaded_file.save(file_path)
                file_paths.append(file_path)

            # Start data collection using the uploaded images
            data_collector = DataCollector(user_id=user_id, name=name, file_uploads=file_paths)
            data_collector.start_collection()

            # After processing, delete the uploaded images
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.remove(file_path)
        else:
            # Start data collection from the webcam
            data_collector = DataCollector(user_id=user_id, name=name)
            data_collector.start_collection()

        flash("Data collection complete", "success")
        return redirect(url_for('home'))

    return render_template('data_collection.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        # Start model training
        trainer = FaceTrainer()
        trainer.train()
        flash("Model training complete", "success")
        return redirect(url_for('home'))

    return render_template('train_model.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        # Load names and start the recognition process
        name_dict = load_names()
        recognizer = FaceRecognizer(name_dict=name_dict)  # Pass the name_dict instead of name_list
        recognizer.recognize()
        flash("Recognition process started", "success")
        return redirect(url_for('home'))

    return render_template('recognize.html')

@app.route('/people')
def people():
    users = User.query.all()
    return render_template('people.html', users=users)

@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.filter_by(user_id=user_id).first()
    if user:
        # Delete user from database
        db.session.delete(user)
        db.session.commit()

        # Delete user's dataset folder
        dataset_folder = os.path.join(app.root_path, 'static', 'datasets', f'{user.name}_{user.user_id}')
        if os.path.exists(dataset_folder):
            shutil.rmtree(dataset_folder)

        flash(f"User {user.name} (ID: {user.user_id}) has been deleted.", "success")
    else:
        flash("User not found.", "danger")
    return redirect(url_for('people'))



if __name__ == '__main__':
    app.run(debug=True)
