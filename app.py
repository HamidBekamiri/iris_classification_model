import os
import joblib
import pandas as pd
import numpy as np
import requests
import zipfile
import streamlit as st

# Define the name of the artifact to download
artifact_name = 'my-model'

# Set up the API endpoint and headers
api_url = f'https://api.github.com/repos/HamidBekamiri/iris_classification_model/actions/artifacts'
headers = {'Accept': 'application/vnd.github.v3+json', 'Authorization': 'token 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'}

# Get the list of artifacts for the repository
response = requests.get(api_url, headers=headers)
response_json = response.json()
artifacts = response_json['artifacts']

# Find the ID of the artifact that matches the specified name
artifact_id = None
for artifact in artifacts:
    if artifact['name'] == artifact_name:
        artifact_id = artifact['id']
        break

# Download the artifact and extract the trained model
if artifact_id:
    artifact_url = f'{api_url}/{artifact_id}/zip'
    response = requests.get(artifact_url, headers=headers)
    with open(f'{artifact_name}.zip', 'wb') as f:
        f.write(response.content)

    with zipfile.ZipFile(f'{artifact_name}.zip', 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.joblib'):
                zip_ref.extract(file)
                model_path = file

    # Load the trained model
    model = joblib.load(model_path)

    # Load the test dataset
    test_data = pd.read_csv("test_data.csv")

    # Extract features and target
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

    # Make predictions on test dataset
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = np.mean(y_pred == y_test)

    # Display results in Streamlit app
    st.write(f"Accuracy: {accuracy}")
else:
    st.write(f"No artifact found with name {artifact_name}")
