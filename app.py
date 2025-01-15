import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Setting up page configuration
st.set_page_config(
    page_title="SVR Charge Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load dataset
data_url = "https://raw.githubusercontent.com/arjunatriananda/data-mining/refs/heads/main/regression.csv"
data = pd.read_csv(data_url)

# Encoding categorical columns
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()
data['sex'] = le_sex.fit_transform(data['sex'])
data['smoker'] = le_smoker.fit_transform(data['smoker'])
data['region'] = le_region.fit_transform(data['region'])

# Preprocess data (split and scale) globally
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR model globally
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)
y_pred = svr.predict(X_test_scaled)

# Sidebar navigation
with st.sidebar:
    st.title("📊 Navigation")
    page = st.radio(
        "Choose a section",
        ["Introduction", "Dataset", "Model Training", "Visualizations", "Predict Charges"]
    )
    st.write("---")

# Title and introduction
if page == "Introduction":
    st.title("📈 Support Vector Regression Charge Predictor")
    st.write("""
    Welcome to the **Support Vector Regression (SVR)** application for medical charge prediction!  
    This interactive tool allows you to explore, visualize, and predict charges using machine learning techniques.
    
    - 📂 **Dataset:** Analyze the dataset used in the prediction.
    - 🚀 **Model Training:** Learn how the SVR model is built and evaluate its performance.
    - 📊 **Visualizations:** Explore interactive charts and learning curves.
    - 🔮 **Predict Charges:** Predict medical charges based on user input.  
    """)

# Dataset exploration
if page == "Dataset":
    st.title("📂 Dataset Overview")
    st.write("### Medical Insurance Charges Dataset")
    if st.checkbox("👀 Show dataset preview"):
        st.dataframe(data.head())
    if st.checkbox("📊 Show dataset statistics"):
        st.write(data.describe())
    st.write("---")
    st.write("### Correlation Heatmap")
    corr_matrix = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

# Preprocessing and Model Training
if page == "Model Training":
    st.title("🚀 Model Training and Evaluation")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success(f"**Model Trained Successfully!**")
    st.write(f"🔢 **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"📈 **R² Score:** {r2:.2f}")

# Visualizations
if page == "Visualizations":
    st.title("📊 Visualizations")
    
    # Sidebar input to select features to visualize
    st.write("### Select the features for testing")
    test_feature = st.selectbox("Choose the feature to visualize:",
                               ['Age', 'BMI', 'Children', 'Smoker', 'Region'])

    # Filter the data based on the selected feature
    if test_feature == 'Age':
        feature_data = X_test['age']
        st.write("### Age vs Charges and Other Features")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot for Age vs Charges
        ax[0, 0].scatter(feature_data, y_test, color="blue", label="Age vs Charges")
        ax[0, 0].set_xlabel("Age")
        ax[0, 0].set_ylabel("Charges")
        ax[0, 0].set_title("Age vs Charges")
        
        # Scatter plot for Age vs BMI
        ax[0, 1].scatter(feature_data, X_test['bmi'], color="green", label="Age vs BMI")
        ax[0, 1].set_xlabel("Age")
        ax[0, 1].set_ylabel("BMI")
        ax[0, 1].set_title("Age vs BMI")

        # Scatter plot for Age vs Children
        ax[1, 0].scatter(feature_data, X_test['children'], color="orange", label="Age vs Children")
        ax[1, 0].set_xlabel("Age")
        ax[1, 0].set_ylabel("Children")
        ax[1, 0].set_title("Age vs Children")

        # Scatter plot for Age vs Smoker
        ax[1, 1].scatter(feature_data, X_test['smoker'], color="purple", label="Age vs Smoker")
        ax[1, 1].set_xlabel("Age")
        ax[1, 1].set_ylabel("Smoker")
        ax[1, 1].set_title("Age vs Smoker")

        # Remove empty subplot area (if any)
        fig.tight_layout(pad=3.0)

        for a in ax.flatten():
            a.legend()

        st.pyplot(fig)

    elif test_feature == 'BMI':
        feature_data = X_test['bmi']
        st.write("### BMI vs Charges and Other Features")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot for BMI vs Charges
        ax[0, 0].scatter(feature_data, y_test, color="blue", label="BMI vs Charges")
        ax[0, 0].set_xlabel("BMI")
        ax[0, 0].set_ylabel("Charges")
        ax[0, 0].set_title("BMI vs Charges")
        
        # Scatter plot for BMI vs Age
        ax[0, 1].scatter(feature_data, X_test['age'], color="red", label="BMI vs Age")
        ax[0, 1].set_xlabel("BMI")
        ax[0, 1].set_ylabel("Age")
        ax[0, 1].set_title("BMI vs Age")

        # Scatter plot for BMI vs Children
        ax[1, 0].scatter(feature_data, X_test['children'], color="orange", label="BMI vs Children")
        ax[1, 0].set_xlabel("BMI")
        ax[1, 0].set_ylabel("Children")
        ax[1, 0].set_title("BMI vs Children")

        # Scatter plot for BMI vs Smoker
        ax[1, 1].scatter(feature_data, X_test['smoker'], color="purple", label="BMI vs Smoker")
        ax[1, 1].set_xlabel("BMI")
        ax[1, 1].set_ylabel("Smoker")
        ax[1, 1].set_title("BMI vs Smoker")

        # Remove empty subplot area (if any)
        fig.tight_layout(pad=3.0)

        for a in ax.flatten():
            a.legend()

        st.pyplot(fig)

    elif test_feature == 'Children':
        feature_data = X_test['children']
        st.write("### Children vs Charges and Other Features")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot for Children vs Charges
        ax[0, 0].scatter(feature_data, y_test, color="blue", label="Children vs Charges")
        ax[0, 0].set_xlabel("Children")
        ax[0, 0].set_ylabel("Charges")
        ax[0, 0].set_title("Children vs Charges")
        
        # Scatter plot for Children vs Age
        ax[0, 1].scatter(feature_data, X_test['age'], color="red", label="Children vs Age")
        ax[0, 1].set_xlabel("Children")
        ax[0, 1].set_ylabel("Age")
        ax[0, 1].set_title("Children vs Age")

        # Scatter plot for Children vs BMI
        ax[1, 0].scatter(feature_data, X_test['bmi'], color="green", label="Children vs BMI")
        ax[1, 0].set_xlabel("Children")
        ax[1, 0].set_ylabel("BMI")
        ax[1, 0].set_title("Children vs BMI")

        # Scatter plot for Children vs Smoker
        ax[1, 1].scatter(feature_data, X_test['smoker'], color="purple", label="Children vs Smoker")
        ax[1, 1].set_xlabel("Children")
        ax[1, 1].set_ylabel("Smoker")
        ax[1, 1].set_title("Children vs Smoker")

        # Remove empty subplot area (if any)
        fig.tight_layout(pad=3.0)

        for a in ax.flatten():
            a.legend()

        st.pyplot(fig)

    elif test_feature == 'Smoker':
        feature_data = X_test['smoker']
        st.write("### Smoker vs Charges and Other Features")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot for Smoker vs Charges
        ax[0, 0].scatter(feature_data, y_test, color="blue", label="Smoker vs Charges")
        ax[0, 0].set_xlabel("Smoker (0 = No, 1 = Yes)")
        ax[0, 0].set_ylabel("Charges")
        ax[0, 0].set_title("Smoker vs Charges")
        
        # Scatter plot for Smoker vs Age
        ax[0, 1].scatter(feature_data, X_test['age'], color="red", label="Smoker vs Age")
        ax[0, 1].set_xlabel("Smoker")
        ax[0, 1].set_ylabel("Age")
        ax[0, 1].set_title("Smoker vs Age")

        # Scatter plot for Smoker vs BMI
        ax[1, 0].scatter(feature_data, X_test['bmi'], color="green", label="Smoker vs BMI")
        ax[1, 0].set_xlabel("Smoker")
        ax[1, 0].set_ylabel("BMI")
        ax[1, 0].set_title("Smoker vs BMI")

        # Scatter plot for Smoker vs Children
        ax[1, 1].scatter(feature_data, X_test['children'], color="orange", label="Smoker vs Children")
        ax[1, 1].set_xlabel("Smoker")
        ax[1, 1].set_ylabel("Children")
        ax[1, 1].set_title("Smoker vs Children")

        # Remove empty subplot area (if any)
        fig.tight_layout(pad=3.0)

        for a in ax.flatten():
            a.legend()

        st.pyplot(fig)

    elif test_feature == 'Region':
        feature_data = X_test['region']
        st.write("### Region vs Charges and Other Features")
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Scatter plot for Region vs Charges
        ax[0, 0].scatter(feature_data, y_test, color="blue", label="Region vs Charges")
        ax[0, 0].set_xlabel("Region")
        ax[0, 0].set_ylabel("Charges")
        ax[0, 0].set_title("Region vs Charges")
        
        # Scatter plot for Region vs Age
        ax[0, 1].scatter(feature_data, X_test['age'], color="red", label="Region vs Age")
        ax[0, 1].set_xlabel("Region")
        ax[0, 1].set_ylabel("Age")
        ax[0, 1].set_title("Region vs Age")

        # Scatter plot for Region vs BMI
        ax[1, 0].scatter(feature_data, X_test['bmi'], color="green", label="Region vs BMI")
        ax[1, 0].set_xlabel("Region")
        ax[1, 0].set_ylabel("BMI")
        ax[1, 0].set_title("Region vs BMI")

        # Scatter plot for Region vs Children
        ax[1, 1].scatter(feature_data, X_test['children'], color="orange", label="Region vs Children")
        ax[1, 1].set_xlabel("Region")
        ax[1, 1].set_ylabel("Children")
        ax[1, 1].set_title("Region vs Children")

        # Remove empty subplot area (if any)
        fig.tight_layout(pad=3.0)

        for a in ax.flatten():
            a.legend()

        st.pyplot(fig)


    # Actual vs Predicted Charges
    st.write("---")
    st.write("### Actual vs Predicted Charges")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color="green")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    ax.set_title("Actual vs Predicted Charges")
    st.pyplot(fig)



# Predict Charges
if page == "Predict Charges":
    st.title("🔮 Predict Charges")
    st.write("### Enter Your Information:")
    
    # Input fields
    age = st.slider("Age", int(data['age'].min()), int(data['age'].max()), 30)
    sex = st.selectbox("Sex", le_sex.classes_)
    bmi = st.slider("BMI", float(data['bmi'].min()), float(data['bmi'].max()), 25.0)
    children = st.slider("Children", int(data['children'].min()), int(data['children'].max()), 0)
    smoker = st.selectbox("Smoker", le_smoker.classes_)
    region = st.selectbox("Region", le_region.classes_)
    
    # Button to trigger prediction
    if st.button("Predict"):
        # Encode user input
        input_data = np.array([[age, le_sex.transform([sex])[0], bmi, children, le_smoker.transform([smoker])[0], le_region.transform([region])[0]]])
        input_data_scaled = scaler.transform(input_data)
        prediction = svr.predict(input_data_scaled)

        # Display the result
        st.write(f"### 🎉 Predicted Charges: **{prediction[0]:.2f}**")
        st.balloons()

