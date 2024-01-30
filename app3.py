import streamlit as st
import joblib
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and vectorizer
try:
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Streamlit app
banner_image_url = 'https://github.com/LNJAMBI001/Streamlit2/blob/main/header_2.jpg?raw=true'  
st.image(banner_image_url, use_column_width=True)
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

# Product options
products = ['Tusker', 'Guinness', 'Gilbeys', 'Captain Morgan', 'Chrome', 'Marketing campaign-OKTOBAFEST','Marketing campaign-WalkerTown']

@st.cache(allow_output_mutation=True)
def load_model_and_vectorizer():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def main():
    st.title('EABL Sentiment Analysis App')
    
    # Dropdown to select product
    selected_product = st.selectbox('Select a product:', products)
    
    st.markdown('<style>textarea{border: 1px solid #3498db;}</style>', unsafe_allow_html=True)
    text = st.text_area('Enter your sentiment about {}:'.format(selected_product), '')

    if st.button('Analyze', key="analyze-button"):
        if text.strip() == '':
            st.warning("Please enter some text before analyzing.")
            return
        model, vectorizer = load_model_and_vectorizer()
        sentiment = predict_sentiment(text, model, vectorizer)
        st.write("The sentiment prediction indicates whether the sentiment expressed about the selected product/event is positive or negative.")
        st.write('Sentiment about {}:'.format(selected_product), sentiment)
        st.write('Your sentiment matters, thank you')

def predict_sentiment(text, model, vectorizer):
    # Preprocess the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    # Ensure the input data is in the appropriate format for prediction
    dtest = xgb.DMatrix(text_vectorized)

    # Use the loaded model to predict sentiment
    sentiment = model.predict(dtest)[0]

    # Adjust this based on your specific model and preprocessing steps
    return 'Negative' if sentiment == 1 else 'Positive'

if __name__ == '__main__':
    main()
