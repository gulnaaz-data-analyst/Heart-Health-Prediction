from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Load the LSTM model for time-series prediction
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, input_shape=(13, 1), activation='relu'))
lstm_model.add(Dense(units=1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the CNN model for X-ray scanning
cnn_model = load_model('model.h5')

# Load the dataset for logistic regression
dataset = pd.read_csv('heart.csv')
X_lr = dataset.drop('output', axis=1)
y_lr = dataset['output']
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
lr_model.fit(X_train_lr, y_train_lr)

# Load the dataset for LSTM
dataset_lstm = pd.read_csv('heart.csv')  # Replace with your actual LSTM dataset
required_columns_lstm = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']  # Replace with the actual column names

# Check if required columns are present in the LSTM dataset
if not set(required_columns_lstm).issubset(dataset_lstm.columns):
    raise ValueError("Required columns are not present in the LSTM dataset.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form for logistic regression model
    user_input_lr = {
        'age': int(request.form['age']),
        'sex': int(request.form['sex']),
        'cp': int(request.form['cp']),
        'trtbps': int(request.form['trtbps']),
        'chol': int(request.form['chol']),
        'fbs': int(request.form['fbs']),
        'restecg': int(request.form['restecg']),
        'thalachh': int(request.form['thalachh']),
        'exng': int(request.form['exng']),
        'oldpeak': float(request.form['oldpeak']),
        'slp': int(request.form['slp']),
        'caa': int(request.form['caa']),
        'thall': int(request.form['thall'])
    }

    # Making predictions using the logistic regression model
    lr_input = [
        user_input_lr['age'], user_input_lr['sex'], user_input_lr['cp'], user_input_lr['trtbps'],
        user_input_lr['chol'], user_input_lr['fbs'], user_input_lr['restecg'], user_input_lr['thalachh'],
        user_input_lr['exng'], user_input_lr['oldpeak'], user_input_lr['slp'], user_input_lr['caa'],
        user_input_lr['thall']
    ]
    lr_prediction = lr_model.predict([lr_input])[0]

    # Making predictions using the LSTM model for time-series data
    lstm_input = np.array([
        user_input_lr['age'], user_input_lr['sex'], user_input_lr['cp'],
        user_input_lr['trtbps'], user_input_lr['chol'], user_input_lr['fbs'],
        user_input_lr['restecg'], user_input_lr['thalachh'], user_input_lr['exng'],
        user_input_lr['oldpeak'], user_input_lr['slp'], user_input_lr['caa'],
        user_input_lr['thall']
    ]).reshape(1, len(required_columns_lstm), 1)
    lstm_prediction = lstm_model.predict(lstm_input)[0][0]

    # Making predictions using the CNN model for X-ray scanning
    xray_image = request.files['xray_image']
    xray_img_path = 'uploads/xray_image.jpg'
    xray_image.save(xray_img_path)

    img = image.load_img(xray_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    cnn_prediction = cnn_model.predict(img_array)[0]

    if (lr_prediction != 0) and (lr_prediction != 1):
        difficulty_heart_disease = True
    else:
        difficulty_heart_disease = False

    # Check for difficulty in X-ray scan prediction
    if (cnn_prediction < 0.0) or (cnn_prediction > 1.0):
        difficulty_xray_scan = True
    else:
        difficulty_xray_scan= False

    overall_result = 'You should go and see a doctor' if (lr_prediction == 1 and cnn_prediction <= 0.5) or (lr_prediction == 0 and cnn_prediction > 0.5) or (lr_prediction == 1 and cnn_prediction > 0.5) else 'You are fine. Have a good day!'

    result = {
        'heart_disease_prediction': 'The patient is likely to have heart disease.' if lr_prediction == 1 else 'The patient is unlikely to have heart disease.',
        'lstm_prediction':  ('Positive' if lstm_prediction > 0.5 else 'Negative'),
        'xray_scan_prediction': 'Abnormal X-ray' if cnn_prediction > 0.5 else 'Normal X-ray',
        'difficulty_heart_disease': difficulty_heart_disease,
        'difficulty_xray_scan': difficulty_xray_scan,
        'overall_result': overall_result
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
