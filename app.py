import streamlit as st
import pandas as pd
import importlib.util
import os
import traceback

st.set_page_config(page_title="Load Forecasting Dashboard", layout="wide")
st.title("⚡ Load Forecasting with VMD + ML Models")

# ------------------------------------------------
# Safe Import Function
# ------------------------------------------------
def safe_import(module_name, file_name):
    try:
        path = os.path.join(os.path.dirname(__file__), file_name)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        st.sidebar.success(f"✅ Loaded {file_name}")
        return module
    except Exception as e:
        st.sidebar.error(f"❌ Could not import {file_name}: {e}")
        st.sidebar.text(traceback.format_exc())
        return None

# ------------------------------------------------
# Import your local scripts safely
# ------------------------------------------------
vmd = safe_import("vmd", "vmd.py")
spm = safe_import("spm", "copy_of_spm_project 1.py")

# ------------------------------------------------
# UI Layout
# ------------------------------------------------
uploaded = st.file_uploader("📂 Upload your dataset (.xlsx)", type=['xlsx'])

if uploaded is not None:
    df = pd.read_excel(uploaded)
    st.success("✅ Dataset uploaded successfully!")
    st.dataframe(df.head())

    if st.button("▶ Run VMD Preprocessing"):
        st.info("Running VMD pipeline ...")
        try:
            if vmd and hasattr(vmd, 'run_vmd_pipeline'):
                result_df = vmd.run_vmd_pipeline(df)
                st.success("✅ VMD completed!")
                st.dataframe(result_df.head())
            else:
                st.warning("⚠ No function 'run_vmd_pipeline' found in vmd.py")
        except Exception as e:
            st.error(f"Error in VMD pipeline: {e}")
            st.text(traceback.format_exc())

    if st.button("🤖 Train ML Models"):
        st.info("Training models, please wait ⏳")
        try:
            if spm and hasattr(spm, 'train_all_models'):
                result = spm.train_all_models(df)
                st.success("✅ Training completed!")
                st.write(result)
            else:
                st.warning("⚠ No function 'train_all_models' found in copy_of_spm_project 1.py")
        except Exception as e:
            st.error(f"Error during training: {e}")
            st.text(traceback.format_exc())
else:
    st.warning("⚠ Please upload your dataset first.")

