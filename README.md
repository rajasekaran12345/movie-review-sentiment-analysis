# 🎬 IMDB Sentiment Analysis with RNN Models

This is a open source project implements a **movie review sentiment analysis system** using the IMDB dataset.  
The app is built with **TensorFlow/Keras** for model training and **Streamlit** for the interactive web interface.  

---

## ✨ Features
- Three deep learning models for classification:
  - **SimpleRNN**
  - **LSTM**
  - **Bidirectional LSTM**
- Interactive **Streamlit UI** with:
  - Right-side floating model selector
  - Dynamic review input box (stored in session state)
  - Sentiment prediction with a single click
- Styled interface with gradient background, floating panel, and colorful buttons  

---

## 📊 Dataset
We use the **IMDB movie review dataset** from `keras.datasets.imdb`,  
which contains 25,000 labeled training reviews and 25,000 testing reviews.  

Each review is preprocessed into word indices and padded to a fixed length of 500 tokens.  

---

## 🧠 Models
All models are trained on the IMDB dataset with **binary sentiment classification (positive/negative)**:

- **SimpleRNN**
  - Layers: Embedding → SimpleRNN → Dense(Sigmoid)
- **LSTM**
  - Layers: Embedding → LSTM → Dense(Sigmoid)
- **BiLSTM**
  - Layers: Embedding → Bidirectional(LSTM) → Dense(Sigmoid)

All models output a probability between `0` (Negative) and `1` (Positive).  

---

## ⚡ Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/imdb-sentiment-rnn.git
   cd imdb-sentiment-rnn
2. Install dependencies:
      pip install -r requirements.txt
3. Train or load the pre-trained models (simplernn_imdb.h5, lstm_imdb.h5, bilstm_imdb.h5).
4. Run the Streamlit app in the terminal: streamlit run main.py.
5. Enter a review, choose a model, and click Classify.

🎨 Example Reviews

Positive:
“The movie was absolutely fantastic! The performances were brilliant, and the story kept me hooked till the end.”

Negative:
“The movie was extremely boring and predictable. The acting felt weak, and I struggled to sit through the entire film.”


📌 Future Improvements

Add pretrained Transformer models (BERT, DistilBERT)

Enhance UI with sentiment visualization (pie/bar chart)

Deploy on Streamlit Cloud or Hugging Face Spaces

🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.