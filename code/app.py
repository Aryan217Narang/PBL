import streamlit as st
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Import your existing modules
from preprocessing import load_and_preprocess
from model import build_cnn_model
from attacks import generate_jsma, generate_fgsm, generate_cw
from defense import pgd_adversarial_training, spatial_smoothing, pioa_optimize
from evaluation import evaluate_model, compute_specificity
from consistency import detect_adversarial

# Page Configuration
st.set_page_config(page_title="Adversarial NIDS Defense", layout="wide")

# Session State Initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🛡️ Hybrid Adversarial Attack Detection & Defense")
st.markdown("""
This dashboard demonstrates the robustness of a DL-based Network Intrusion Detection System (NIDS) 
against adversarial perturbations using the Hybrid (PGD + PIOA + SS) defense strategy.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Pipeline Configuration")
dataset_option = st.sidebar.selectbox("Select Dataset", ["CIC-IDS2017"])
rows_to_load = st.sidebar.slider("Rows per file", 1000, 20000, 5000)
epochs = st.sidebar.number_input("Training Epochs", 1, 50, 5)

# Setup file paths (Make sure these folders/files exist in your project)
csv_files = [
    "data/CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv",
    "data/CIC-IDS2017/Tuesday-WorkingHours.pcap_ISCX.csv"
]

# --- 1. Preprocessing & Training ---
if st.sidebar.button("🚀 Run Preprocessing & Train Base Model"):
    with st.status("Processing Data...", expanded=True) as status:
        st.write("Loading datasets and applying ICA/RFE...")
        X_train, X_test, y_train, y_test, n_classes = load_and_preprocess(
            file_paths=csv_files,
            rows_per_file=rows_to_load
        )
        
        # Limit test set for browser performance
        X_test, y_test = X_test[:500], y_test[:500]
        st.session_state.data = (X_train, X_test, y_train, y_test, n_classes)
        
        st.write("Building CNN Model...")
        model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)
        
        st.write("Training model (this may take a minute)...")
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=32, 
            validation_split=0.1, 
            verbose=0
        )
        st.session_state.model = model
        status.update(label="Training Complete!", state="complete")

# --- Main Dashboard ---
if st.session_state.model and st.session_state.data:
    X_train, X_test, y_train, y_test, n_classes = st.session_state.data
    model = st.session_state.model

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Base Model Performance")
        res_pre = evaluate_model(model, X_test, y_test, label="Pre-Attack")
        st.metric("Accuracy", f"{res_pre['accuracy']}%")
        st.metric("F1-Score", f"{res_pre['f1_score']}%")
        st.write(f"**Confusion Matrix (Clean):**")
        st.write(res_pre['cm'])

    # --- 2. Adversarial Attacks ---
    st.divider()
    st.header("⚔️ Adversarial Attack Simulation")
    
    attack_type = st.radio("Select Attack to Generate", ["FGSM", "JSMA", "C&W"], horizontal=True)
    epsilon = st.slider("Epsilon (Perturbation Strength)", 0.01, 0.1, 0.03)
    
    if st.button("🔥 Launch Attack"):
        params = {"epsilon": epsilon, "max_iter": 10}
        
        with st.spinner(f"Generating {attack_type} perturbations..."):
            if attack_type == "FGSM":
                X_adv = generate_fgsm(model, X_test, epsilon)
            elif attack_type == "JSMA":
                X_adv = generate_jsma(model, X_test, y_test, params)
            else:
                X_adv = generate_cw(model, X_test, y_test, params)
        
        res_adv = evaluate_model(model, X_adv, y_test, label="Post-Attack", X_clean=X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Post-Attack Accuracy", f"{res_adv['accuracy']}%", delta=f"{res_adv['accuracy'] - res_pre['accuracy']}%", delta_color="inverse")
        c2.metric("Attack Success Rate (ASR)", f"{res_adv['asr']}%")
        c3.metric("F1-Score", f"{res_adv['f1_score']}%")

        # --- 3. Defense & Detection ---
        st.divider()
        st.header("🛡️ Defense Performance")
        
        # AICC + TCC Detection
        st.subheader("1. Consistency Detection (AICC + TCC)")
        with st.spinner("Running Consistency Checks..."):
            adv_flags = detect_adversarial(model, X_adv[:100], window_size=3)
            # Assuming all samples here are adversarial for the demo
            det_rate = (np.sum(adv_flags) / len(adv_flags)) * 100
            st.success(f"Adversarial Detection Rate: **{det_rate:.2f}%**")
            st.info("Inputs failing the consistency check are blocked before reaching the target model.")

        # Spatial Smoothing Comparison
        st.subheader("2. Spatial Smoothing Defense")
        X_ss = spatial_smoothing(X_adv, window_radius=1.5)
        res_ss = evaluate_model(model, X_ss, y_test, label="SS Defense")
        st.metric("Accuracy after SS", f"{res_ss['accuracy']}%", delta=f"{res_ss['accuracy'] - res_adv['accuracy']}%")

else:
    st.info("Please use the sidebar to load data and train the model to start the simulation.")