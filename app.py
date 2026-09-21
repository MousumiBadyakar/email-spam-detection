import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Load model and supporting files
@st.cache_resource
def load_model_files():

    model = load_model("gru_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    with open("label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    return model, tokenizer, config, label_mapping


model, tokenizer, config, label_mapping = load_model_files()


# Text preprocessing
def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stops = set(stopwords.words("english"))
    words = [word for word in words if word not in stops]

    # Join words
    text = " ".join(words)

    # Remove hyperlinks
    text = re.sub(r"http\S+", "", text)

    return text


# Prediction function
def predict_email(email):

    cleaned_email = preprocess_text(email)

    sequence = tokenizer.texts_to_sequences([cleaned_email])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=config["max_length"],
        padding="post"
    )

    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    predicted_class = int(probability > 0.5)

    label = label_mapping[predicted_class]

    return label, probability


# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


# UI
st.title("📧 Email Spam Detector")

st.write(
    "Enter an email message below to check whether it is Spam or Ham."
)


email_text = st.text_area(
    "Enter your email:",
    height=220,
    placeholder="Paste your email message here..."
)


if st.button("🔍 Check Email", use_container_width=True):

    if email_text.strip() == "":
        st.warning("Please enter an email message.")

    else:

        label, probability = predict_email(email_text)

        if label == "Spam":

            st.error("🚨 SPAM EMAIL")

            st.write(
                f"Spam Probability: **{probability * 100:.2f}%**"
            )

        else:

            st.success("✅ HAM / NOT SPAM")

            st.write(
                f"Ham Probability: **{(1 - probability) * 100:.2f}%**"
            )