import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Data Collection & Pre-processing 
# Loading the data from csv file to a pandas DataFrame
mail_data = pd.read_csv('mail_data.csv')

st.set_page_config(
    page_title="Spam Email Detection",  # This is the title that will appear in the browser tab
            # Optional: You can set a custom favicon
    layout="wide"                     # Optional: Set layout to "wide" or "centered"
)


# Replacing all null values with a null string
mail_data2 = mail_data.where((pd.notnull(mail_data)), '')

# Label Encoding: Labeling spam mail as 0; ham mail as 1
mail_data2.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data2.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the data as text and label
X = mail_data2['Message']
Y = mail_data2['Category'].astype(int)

# Splitting data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature Extraction: Transforming the text data to feature vectors
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Training the Logistic Regression model with the training data
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluating the model
accuracy_on_training_data = accuracy_score(Y_train, model.predict(X_train_features))
accuracy_on_test_data = accuracy_score(Y_test, model.predict(X_test_features))

# Save the model and vectorizer using pickle
pickle.dump(model, open('spam_detection_model.pkl', 'wb'))
pickle.dump(feature_extraction, open('tfidf_vectorizer.pkl', 'wb'))

# Streamlit UI
st.title("Spam Detection System")

# Input text box for the user to enter an email message
message = st.text_area("Enter the Content of Email:")

# Function to predict whether the message is spam or ham
def predict_spam(message):
    # Load the model and vectorizer
    model = pickle.load(open('spam_detection_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    
    # Transform the message using the vectorizer
    transformed_message = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(transformed_message)[0]
    
    return prediction

# Button to classify the message
if st.button("Classify"):
    if message.strip() == "":
        st.error("Please enter a message to classify.")
    else:
        # Make the prediction
        prediction = predict_spam(message)
        
        # Display the result
        if prediction == 0:
            st.error("It's a Spam Email.")
        else:
            st.success("It's a Ham Email.")

# Additional details for the user
st.write("""
### About the Spam Detection System
This application uses a Logistic Regression model to classify emails as either Spam or Ham. 
It leverages a TF-IDF vectorizer to transform the input email content into numerical features 
that the model can process for classification.
""")
