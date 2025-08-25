# Step 1: Import Libraries and Load the Models
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained models
lstm_model = load_model('lstm_imdb.keras')
simplernn_model = load_model('simple_rnn_imdb.keras')
bilstm_model = load_model('bilstm_imdb.keras')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #f6d365, #fda085);
        font-family: 'Arial', sans-serif;
    }
    .right-panel {
        position: fixed;
        top: 35%;
        right: 30px;
        width: 250px;
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-10px); }
        100% { transform: translatey(0px); }
    }
    .model-btn {
        flex: 1;
        margin: 5px;
        padding: 6px 10px;
        font-size: 14px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: 0.3s;
        color: white;
    }
    .simplernn { background-color: #6dd5ed; }
    .lstm { background-color: #2193b0; }
    .bilstm { background-color: #cc2b5e; }
    .selected { border: 2px solid black; box-shadow: 0px 0px 10px black; }
    .model-btn:hover { opacity: 0.85; transform: scale(1.05); }
    </style>
""", unsafe_allow_html=True)

# Streamlit App
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review and choose a model to classify it.')

# Input box (stores dynamically in session state)
st.session_state.user_input = st.text_area(
    '📝 Movie Review', 
    value=st.session_state.user_input, 
    key="review_input"
)

# Right panel with styled buttons in a single row
st.markdown('<div class="right-panel"> <h4>⚡ Choose your AI Model</h4>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(("▶ " if st.session_state.selected_model == "simplernn" else "") + "SimpleRNN"):
        st.session_state.selected_model = "simplernn"

with col2:
    if st.button(("▶ " if st.session_state.selected_model == "lstm" else "") + "LSTM"):
        st.session_state.selected_model = "lstm"

with col3:
    if st.button(("▶ " if st.session_state.selected_model == "bilstm" else "") + "BiLSTM"):
        st.session_state.selected_model = "bilstm"

st.markdown('</div>', unsafe_allow_html=True)

# Classify button (separate)
if st.button("🚀 Classify"):
    if st.session_state.user_input and st.session_state.selected_model:
        if st.session_state.selected_model == "simplernn":
            model = simplernn_model
        elif st.session_state.selected_model == "lstm":
            model = lstm_model
        elif st.session_state.selected_model == "bilstm":
            model = bilstm_model

        with st.spinner("🔍 Analyzing..."):
            preprocessed_input = preprocess_text(st.session_state.user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = '🌟 Positive' if prediction[0][0] > 0.5 else '💔 Negative'

        st.success(f'Sentiment: {sentiment}')
        st.write(f'🔢 Prediction Score: {prediction[0][0]:.4f}')
    else:
        st.warning("⚠ Please enter a review and select a model.")
