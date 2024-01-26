import streamlit as st
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
model = joblib.load('best_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
banner_image_path = r'C:/Users/ADMIN/Downloads/Eabl_black2.png'  
st.image(banner_image_path, use_column_width=True)
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .analyze-button {
            background-color: #3498db;
            color: #ffffff;
            padding: 10px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    st.title('EABL Sentiment Analysis App')
    text = st.text_area('Enter your sentiment:', '')

    if st.button('Analyze', key="analyze-button"):
        sentiment = predict_sentiment(text)
        st.write('Sentiment:', sentiment)


def predict_sentiment(text):
    # Preprocess the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Ensure the input data is in the appropriate format for prediction
    dtest = xgb.DMatrix(text_vectorized)

    # Use the loaded model to predict sentiment
    sentiment = model.predict(dtest)[0]

    # Adjust this based on your specific model and preprocessing steps
    if sentiment == 0:
        return 'Neutral'
    elif sentiment == 1:
        return 'Negative'
    else:
        return 'Positive'


if __name__ == '__main__':
    main()
