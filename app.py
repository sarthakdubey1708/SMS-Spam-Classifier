
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
    page_title="SMS Spam Classifier",
    layout="centered"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 780px;
    }
</style>
""", unsafe_allow_html=True)

model_path = Path(__file__).parent / "spam_classifier.joblib"
saved_data = joblib.load(model_path)

model = saved_data["model"]
vectorizer = saved_data["vectorizer"]

st.title("SMS Spam Classifier")
st.caption("Machine Learning | NLP | Python")

st.write(
    "Detect potentially unwanted SMS messages using "
    "a machine-learning model trained on real SMS data."
)

st.divider()
st.subheader("Analyze a Message")

message = st.text_area(
    "Enter your SMS message",
    placeholder="Type or paste an SMS message here...",
    height=140
)

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == "spam":
            st.error("SPAM DETECTED")
            st.write("This message may be unwanted or promotional.")
        else:
            st.success("LIKELY LEGITIMATE")
            st.write("The model classified this message as ham.")

st.divider()

with st.expander("How does this classifier work?"):
    st.write(
        "1. The SMS is converted into numerical features using TF-IDF."
    )
    st.write(
        "2. A trained Multinomial Naive Bayes model analyzes those features."
    )
    st.write(
        "3. The model predicts whether the message is spam or ham."
    )

st.caption(
    "Predictions are statistical estimates and may occasionally be incorrect."
)
