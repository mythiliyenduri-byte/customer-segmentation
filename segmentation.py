import streamlit as st
import pandas as pd
import numpy as np
import joblib
kmeans=joblib.load("customer_segmentation_model.pkl")
scaler=joblib.load("scaler.pkl")
st.title("Customer Segmentation App")
st.write("enter the customer details to predict the segment")
age=st.number_input("Age",min_value=18,max_value=100,value=35)
income=st.number_input("Income",min_value=0,max_value=1000000,value=50000)
total_spending=st.number_input("Total Spending(sum of purcheses) ",min_value=0,max_value=5000,value=1000)
num_web_purshaes=st.number_input("Number of web purchases",min_value=0,max_value=100,value=10)
num_store_purchases=st.number_input("Number of store purchases",min_value=0,max_value=100,value=10)
num_web_visits=st.number_input("Number of web visits",min_value=0,max_value=50,value=3)
recency=st.number_input("Recency (days since last purchase)",min_value=0,max_value=365,value=30)


input_data = pd.DataFrame([{
    'Age': age,
    'Income': income,
    'Total_spending': total_spending,
    'NumWebPurchases': num_web_purshaes,
    'NumStorePurchases': num_store_purchases,
    'NumWebVisitsMonth': num_web_visits,
    'Recency': recency
}])

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)
    st.success(f"Predicted segment: Cluster {cluster[0]}")

    st.write("""
    - Cluster 0: High budget
    - Cluster 1: High spending
    - Cluster 2: Web visitors
    """)
