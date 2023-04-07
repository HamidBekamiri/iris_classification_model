
import pickle
# import streamlit as st
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    species = model.predict(features)
    return iris.target_names[species[0]]

# Create the Streamlit app
st.title("Iris Species Prediction")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("Predict"):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f"The predicted species is {species}.")
