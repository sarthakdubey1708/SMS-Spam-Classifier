
# SMS Spam Classifier

A machine learning application that classifies SMS messages as spam or legitimate (ham) using Natural Language Processing.

The project uses TF-IDF for text feature extraction and Multinomial Naive Bayes for classification. A Streamlit interface allows users to enter messages and receive predictions.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Streamlit
- Joblib

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

The original dataset contains 5,572 labeled SMS messages. After removing duplicate records, 5,169 messages remain.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Model Performance

The model was trained using Multinomial Naive Bayes with alpha=0.1.

Results from the 1,034-message test split:

| Metric | Result |
|---|---|
| Accuracy | 98.26% |
| Spam Precision | 97.48% |
| Spam Recall | 88.55% |
| Spam F1-score | 92.80% |

The test set was also inspected during development, so these results should not be treated as an entirely untouched benchmark.

## How It Works

1. SMS messages are loaded and duplicate records are removed.
2. The dataset is divided into training and testing sets.
3. TF-IDF converts message text into numerical features.
4. Multinomial Naive Bayes learns patterns associated with spam and ham.
5. The trained model and vectorizer are saved using Joblib.
6. Streamlit loads the saved model and classifies user-entered messages.

## Running the Project

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

The application opens in your browser.

## Files

- `main.py` - Data processing, model training and evaluation
- `app.py` - Streamlit application
- `spam_classifier.joblib` - Saved model and vectorizer
- `requirements.txt` - Python dependencies
- `SMSSpamCollection` - SMS dataset

## Limitations

The classifier may incorrectly classify unfamiliar messages. Predictions depend on patterns learned from the training dataset and should not be treated as guaranteed spam detection.
