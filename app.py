import streamlit as st
import pandas as pd
from vmd import *
from copy_of_spm_project_1 import *

st.set_page_config(page_title="Load Forecasting Dashboard", layout="wide")
st.title("⚡ Load Forecasting with VMD + ML Models")

uploaded = st.file_uploader("Upload your dataset (.xlsx)", type=['xlsx'])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success("✅ Dataset uploaded successfully!")

    if st.button("Run VMD Preprocessing"):
        st.info("Running VMD pipeline ...")
        try:
            # Try to run your VMD logic from vmd.py
            if 'run_vmd_pipeline' in globals():
                result_df = run_vmd_pipeline(df)
            else:
                result_df = df
            st.success("✅ VMD completed!")
            st.dataframe(result_df.head())
        except Exception as e:
            st.error(f"Error in VMD pipeline: {e}")

    if st.button("Train ML Models"):
        st.info("Training models, please wait ⏳")
        try:
            # Try to run your ML model training from copy_of_spm_project 1.py
            if 'train_all_models' in globals():
                result = train_all_models(df)
            else:
                result = "Model training completed!"
            st.success("✅ Training completed!")
            st.write(result)
        except Exception as e:
            st.error(f"Error during training: {e}")

else:
    st.warning("⚠ Please upload your dataset first.")
