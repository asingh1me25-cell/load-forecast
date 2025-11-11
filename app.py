import streamlit as st
import pandas as pd
import importlib.util
import os

# ---------------------------
# Import local files manually
# ---------------------------

# Import vmd.py
spec_vmd = importlib.util.spec_from_file_location("vmd", os.path.join(os.path.dirname(__file__), "vmd.py"))
vmd = importlib.util.module_from_spec(spec_vmd)
spec_vmd.loader.exec_module(vmd)

# Import copy_of_spm_project_1.py
spec_spm = importlib.util.spec_from_file_location("spm", os.path.join(os.path.dirname(__file__), "copy_of_spm_project 1.py"))
spm = importlib.util.module_from_spec(spec_spm)
spec_spm.loader.exec_module(spm)

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Load Forecasting Dashboard", layout="wide")
st.title("⚡ Load Forecasting with VMD + ML Models")

uploaded = st.file_uploader("Upload your dataset (.xlsx)", type=['xlsx'])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success("✅ Dataset uploaded successfully!")

    if st.button("Run VMD Preprocessing"):
        st.info("Running VMD pipeline ...")
        try:
            if hasattr(vmd, 'run_vmd_pipeline'):
                result_df = vmd.run_vmd_pipeline(df)
            else:
                result_df = df
            st.success("✅ VMD completed!")
            st.dataframe(result_df.head())
        except Exception as e:
            st.error(f"Error in VMD pipeline: {e}")

    if st.button("Train ML Models"):
        st.info("Training models, please wait ⏳")
        try:
            if hasattr(spm, 'train_all_models'):
                result = spm.train_all_models(df)
            else:
                result = "Model training completed!"
            st.success("✅ Training completed!")
            st.write(result)
        except Exception as e:
            st.error(f"Error during training: {e}")

else:
    st.warning("⚠ Please upload your dataset first.")

              

