import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------------------------
# APP TITLE
# -----------------------------------------
st.title("🌸 Iris Flower Classification using Decision Tree")

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------
st.subheader("📌 Loading Iris Dataset")

try:
    df = pd.read_csv('/kaggle/input/iris/Iris.csv')
    st.success("Dataset loaded from Kaggle input folder successfully.")
except:
    st.warning("File not found. Loading dataset from sklearn...")
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Species'] = data.target_names[data.target]

# Remove Id column if exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

st.write(df.head())

# -----------------------------------------
# SPLIT DATA
# -----------------------------------------
X = df.drop('Species', axis=1)
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# TRAIN MODEL
# -----------------------------------------
st.subheader("📘 Training Decision Tree Model")

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

st.success("Model trained successfully!")

# -----------------------------------------
# MODEL EVALUATION
# -----------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Accuracy: {accuracy * 100:.2f}%**")

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# -----------------------------------------
# DECISION TREE VISUALIZATION
# -----------------------------------------
st.subheader("🌳 Decision Tree Visualization")

fig = plt.figure(figsize=(12, 8))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=model.classes_,
)
st.pyplot(fig)

# -----------------------------------------
# SAVE MODEL
# -----------------------------------------
with open('decision_tree_iris.pkl', 'wb') as f:
    pickle.dump(model, f)

st.success("Model saved as 'decision_tree_iris.pkl'. You can download it from the Output section.")

# -----------------------------------------
# PREDICTION SECTION
# -----------------------------------------
st.subheader("🔮 Make a Prediction")

sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"🌼 Predicted Species: **{prediction}**")
