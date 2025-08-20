import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("lr_tfidf.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📩 SMS Spam Classifier")
user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    label = "Spam 🚫" if prediction == 1 else "Ham ✅"
    st.write("Prediction:", label)
