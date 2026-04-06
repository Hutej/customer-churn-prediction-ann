import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle


# Load the trained regression model
model = tf.keras.models.load_model("notebook/regression_model.h5")

# Load encoders and scaler
with open("notebook/onehot_encoder_geo.pkl", "rb") as file:
	onehot_encoder_geo = pickle.load(file)

with open("notebook/label_encoder_gender.pkl", "rb") as file:
	label_encoder_gender = pickle.load(file)

with open("notebook/scalar.pkl", "rb") as file:
	scalar = pickle.load(file)


# Streamlit app
st.title("Salary Estimation")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure", 0, 10)
num_of_product = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame(
	{
		"CreditScore": [credit_score],
		"Gender": [label_encoder_gender.transform([gender])[0]],
		"Age": [age],
		"Tenure": [tenure],
		"Balance": [balance],
		"NumOfProducts": [num_of_product],
		"HasCrCard": [has_cr_card],
		"IsActiveMember": [is_active_member],
	}
)

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
	geo_encoded,
	columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
)

# Concatenate encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Align columns with scaler fit-time schema to avoid feature-name mismatch errors.
if hasattr(scalar, "feature_names_in_"):
	missing_cols = [col for col in scalar.feature_names_in_ if col not in input_data.columns]
	for col in missing_cols:
		input_data[col] = 0
	input_data = input_data.reindex(columns=scalar.feature_names_in_, fill_value=0)
	if missing_cols:
		st.caption(f"Auto-filled missing features for scaler: {', '.join(missing_cols)}")

# Scale input data
input_data_scaled = scalar.transform(input_data)

# Predict salary
prediction = model.predict(input_data_scaled, verbose=0)
predicted_salary = prediction[0][0]

st.write(f"Estimated Salary: ${predicted_salary:,.2f}")
