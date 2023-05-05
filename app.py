from flask_sqlalchemy import SQLAlchemy
import io
from PIL import Image
import numpy as np
from flask import Flask, render_template, request,session, redirect, url_for
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user
from flask_bcrypt import Bcrypt
from wtforms import StringField, PasswordField, EmailField,SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the absolute file path of the model
model_path = os.path.join(current_dir, 'model2.tflite')

# Load the trained model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a Flask application
app = Flask(__name__, template_folder='templates')
app.secret_key = 'secret_key'#os.urandom(24)

#Define the database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.app_context().push()

@app.before_first_request
def create_tables():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key = True)
    first_name = db.Column(db.String(200),nullable=False)
    last_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), unique=True, nullable=False)

    def __init__(self,first_name,last_name,email,password):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.password = password

class RegistrationForm(FlaskForm):
    first_name = StringField('first_name', validators={InputRequired(), Length(min=3,max=15)}, render_kw={"placeholder":"First Name"})
    last_name = StringField('last_name', validators={InputRequired(), Length(min=3,max=15)},render_kw={"placeholder":"Last Name"})
    email = EmailField('email',validators={InputRequired(),Length(max=50)},render_kw={"placeholder":"Email"})
    password = PasswordField('password',validators={InputRequired(),Length(min=8,max=15)},render_kw={"placeholder":"Password"})
    submit = SubmitField("Register")

    def validate_email(self,email):
        existing_email = User.query.filter_by(
            email = email.data
        ).first()

        if existing_email:
            raise ValidationError('Email already exists')


class LoginForm(FlaskForm):
    email = EmailField('email',validators={InputRequired(),Length(max=50)},render_kw={"placeholder":"Email"})
    password = PasswordField('password',validators = {InputRequired(),Length(min=8,max=15)}, render_kw={"placeholder":"Password"})
    submit = SubmitField("Login")


@app.route('/index')
@app.route('/')
def index():
    if 'username' in session:
        #return render_template('html/index.html')
        return redirect(url_for('predict'))
    else:
        #return redirect(url_for('login'))
        return render_template('html/index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('predict'))
    return render_template('html/login.html',form = form)

@app.route('/signup')
def signup():
    form = RegistrationForm()
    print('form created')
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(
            first_name = form.first_name.data,
            last_name = form.last_name.data,
            email = form.email.data,
            password = hashed_password
        )
        print('New User',new_user)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('html/signup.html',form = form)



# def preprocess_image(image):
#     # Open the image using PIL
#     img = Image.open(image)

#     # Convert the image to RGB if it's not already
#     if img.mode != 'RGB':
#         img = img.convert('RGB')

#     # Resize the image to match the input size of the model
#     img = img.resize((224, 224))

#     # Convert the image to a numpy array
#     img = np.array(img)

#     # Normalize the image data
#     img = img / 255.0

#     # Expand the dimensions to match the input shape of the model
#     img = np.expand_dims(img, axis=0)

#     return img
def preprocess_image(image, target_size):
    # Convert image to RGB mode if it's not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize the image to the target size
    image = image.resize(target_size)

    # Convert image to numpy array
    image_array = np.array(image)

    # Normalize the image
    image_array = image_array / 255.0

    # Expand the dimensions of the image array to match the model input shape
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Define a route for the prediction API
@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']

        # Read the image file as bytes and convert it to a PIL Image object
        #image_bytes = file.read()
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess the image
        image = preprocess_image(image, target_size=(224,224))

        # Set the input tensor
        input_data = image.astype('float32')
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Perform inference
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Postprocess the output data (modify according to your needs)
        predictions = output_data

        predicted_class = np.argmax(predictions[0])
        accuracy = round(float(predictions[0][predicted_class]), 2)
        if accuracy > 0.5:
            label = "Pneumonia"
        else:
            label = "Not Having Pneumonia"
        #return redirect(url_for('result', accuracy=accuracy*100,label = label))
        return render_template('html/result.html',accuracy=(accuracy*100),label=label)


    return render_template('html/predict.html')



# @app.route('/result')
# @login_required
# def result():
#     # accuracy = request.args.get('accuracy')

#     # if accuracy == "1.0":
#     #     status = "The patient has Pneumonia"
#     # else:
#     #     status = "The patient does not have Pneumonia"
#     return render_template('html/result.html', accuracy=status)


#define a route for logout
@app.route('/logout', methods = ['GET','POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Start the Flask application
if __name__ == '__main__':
    db.create_all()
    app.run(debug=False, host= '0.0.0.0' )#port=int(os.environ.get('PORT', 8080)))
