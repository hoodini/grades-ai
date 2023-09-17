import streamlit as st
import numpy as np
import joblib

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, w1=None, w2=None):
        self.w1 = w1 if w1 is not None else np.random.randn(input_size, hidden_size)
        self.w2 = w2 if w2 is not None else np.random.randn(hidden_size, output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def predict(self, x):
        h = self.relu(x.dot(self.w1))
        y = h.dot(self.w2)
        y = np.clip(y, 0, 100)  # Clip the predictions to be between 0 and 100
        return y

# Load the pre-trained model weights
w1 = np.load('w1.npy')
w2 = np.load('w2.npy')
model = SimpleNN(3, 10, 1, w1, w2)

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title('Exam Score Prediction')
st.write("""
Enter the values to get the exam score prediction.
""")
age = st.number_input('Age', value=25)
gender = st.number_input('Gender (0 for male, 1 for female)', value=0)
sleep = st.number_input('Average Sleeping Hours', value=7)
if st.button('Predict'):
    input_data = np.array([age, gender, sleep]).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Scale the input data
    prediction = model.predict(input_data)[0,0]
    st.write(f'Predicted Exam Score: {prediction:.2f}')
