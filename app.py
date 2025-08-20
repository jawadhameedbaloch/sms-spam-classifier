import streamlit as st
import pickle

# Set page title and favicon
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# Load the saved model and vectorizer
with open("lr_tfidf.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# App title
st.title("📩 SMS Spam Classifier")
st.subheader("Developed by Jawad Hameed Baloch")

# Input text area
user_input = st.text_area("Enter your message:")

# Classify button
if st.button("Classify"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    label = "Spam 🚫" if prediction == 1 else "Ham ✅"
    st.write("Prediction:", label)
