import streamlit as st
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Import existing modules
from preprocessing import load_and_preprocess
from model import build_cnn_model
from attacks import generate_jsma, generate_fgsm, generate_cw
from defense import pgd_adversarial_training, spatial_smoothing, pioa_optimize
from evaluation import evaluate_model, compute_specificity
from consistency import detect_adversarial

# Page Configuration
st.set_page_config(page_title="Adversarial NIDS Defense (CIC-DDoS2019)", layout="wide")

# Session State Initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

st.title("🛡️ Hybrid Adversarial Attack Detection & Defense")
st.markdown("""
This dashboard demonstrates the robustness of a DL-based Network Intrusion Detection System (NIDS) 
against adversarial perturbations on the **CIC-DDoS2019** dataset using the Hybrid (PGD + PIOA + SS + AICC/TCC) defense strategy.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Pipeline Configuration")
dataset_option = st.sidebar.selectbox("Select Dataset", ["CIC-DDoS2019"])
rows_to_load = st.sidebar.slider("Rows per file", 1000, 20000, 5000)
epochs = st.sidebar.number_input("Training Epochs", 1, 50, 5)

def resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    alt = os.path.join("code", p)
    if os.path.exists(alt):
        return alt
    parent_alt = os.path.join("..", p)
    if os.path.exists(parent_alt):
        return parent_alt
    return p

csv_files = [
    resolve_path("data/cic-ids-2019/DrDoS_DNS_data_1_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_LDAP_data_2_0_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_MSSQL_data_1_3_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_NTP_data_data_5_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_NetBIOS_data_1_3_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_SNMP_data_1_3_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_SSDP_data_2_per.csv"),
    resolve_path("data/cic-ids-2019/DrDoS_UDP_data_2_per.csv"),
    resolve_path("data/cic-ids-2019/UDPLag_data_2_0_per.csv"),
    resolve_path("data/cic-ids-2019/syn_data.csv"),
]

# --- 1. Preprocessing & Training ---
if st.sidebar.button("🚀 Run Preprocessing & Train Base Model"):
    with st.status("Processing Data...", expanded=True) as status:
        st.write("Loading CIC-DDoS2019 datasets and applying FastICA/RFE...")
        X_train, X_test, y_train, y_test, n_classes = load_and_preprocess(
            file_paths=csv_files,
            rows_per_file=rows_to_load
        )
        
        # Limit test set for browser performance
        X_test, y_test = X_test[:500], y_test[:500]
        st.session_state.data = (X_train, X_test, y_train, y_test, n_classes)
        
        st.write("Building 1D-CNN Model...")
        model = build_cnn_model(input_shape=X_train.shape[1:], n_classes=n_classes)
        
        st.write("Training model...")
        history = model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=64, 
            validation_split=0.1, 
            verbose=0
        )
        st.session_state.model = model
        status.update(label="Training Complete!", state="complete", expanded=False)

# --- Main Dashboard ---
if st.session_state.model is not None and st.session_state.data is not None:
    X_train, X_test, y_train, y_test, n_classes = st.session_state.data
    model = st.session_state.model
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Pre-Attack Baseline Performance")
        pre_results = evaluate_model(model, X_test, y_test, label="Pre-Attack")
        st.metric(label="Accuracy", value=f"{pre_results['accuracy']:.2f}%")
        st.metric(label="F1 Score", value=f"{pre_results['f1_score']:.2f}%")
        st.metric(label="Specificity", value=f"{pre_results['specificity']:.2f}%")

    with col2:
        st.subheader("2. Run Adversarial Attack")
        attack_type = st.selectbox("Choose Attack Method", ["FGSM", "JSMA", "C&W"])
        epsilon = st.slider("Perturbation (Epsilon / Strength)", 0.01, 0.30, 0.10)
        
        if st.button("Generate Adversarial Attack"):
            params = {"epsilon": epsilon, "max_iter": 15, "sigma": 2.0, "alpha": 0.005}
            with st.spinner("Generating adversarial samples..."):
                if attack_type == "FGSM":
                    X_adv = generate_fgsm(model, X_test, epsilon=epsilon)
                elif attack_type == "JSMA":
                    X_adv = generate_jsma(model, X_test, y_test, params)
                elif attack_type == "C&W":
                    X_adv = generate_cw(model, X_test, y_test, params)
                    
                st.session_state.X_adv = X_adv
                st.session_state.adv_results = evaluate_model(model, X_adv, y_test, label=f"Post-{attack_type}")
            
    if 'adv_results' in st.session_state:
        st.divider()
        st.subheader("3. Post-Attack Degradation vs Hybrid Defense")
        adv_res = st.session_state.adv_results
        X_adv = st.session_state.X_adv
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Under Attack Accuracy", f"{adv_res['accuracy']:.2f}%", delta=f"{adv_res['accuracy'] - pre_results['accuracy']:.2f}%", delta_color="inverse")
        c1.metric("Attack Success Rate (ASR)", f"{adv_res['asr']:.2f}%" if adv_res['asr'] != "N/A" else "N/A")
        
        with c2:
            st.markdown("#### Test-Time Defense: Spatial Smoothing (SS)")
            X_smoothed = spatial_smoothing(X_adv, window_radius=2.0)
            ss_res = evaluate_model(model, X_smoothed, y_test, label="SS Defense")
            st.metric("SS Recovered Accuracy", f"{ss_res['accuracy']:.2f}%", delta=f"{ss_res['accuracy'] - adv_res['accuracy']:.2f}%")
            
        with c3:
            st.markdown("#### Dynamic Consistency Defense: AICC + TCC")
            adv_flags = detect_adversarial(model, X_smoothed, window_size=3, aicc_thresh=0.7, final_thresh=0.75)
            detected_rate = (np.sum(adv_flags) / len(adv_flags)) * 100
            st.metric("Adversarial Detection Rate", f"{detected_rate:.2f}%")
            
            clean_idx = np.where(adv_flags == 0)[0]
            if len(clean_idx) > 0:
                final_res = evaluate_model(model, X_smoothed[clean_idx], y_test[clean_idx], label="Hybrid Defense")
                st.metric("Final Defended Accuracy", f"{final_res['accuracy']:.2f}%")
            else:
                st.write("All perturbed flows identified & blocked.")
else:
    st.info("👈 Please configure and click **Run Preprocessing & Train Base Model** from the sidebar to begin.")