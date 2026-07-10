import os
import json
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# Page Configuration and CSS Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="CreditWise - Loan Approval Intelligent System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI CSS styling (adapted for dark and light modes)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Premium Gradient Background for entire app */
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #0f172a 65%) !important;
    }
    
    /* Styled Glassy Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Root container font family */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Glassmorphic Metrics Card */
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.4);
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 5px 0;
    }
    .kpi-lbl {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 16px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.8s ease-in-out;
    }
    .approved-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%);
        border: 2px solid #22c55e;
        color: #f8fafc;
    }
    .rejected-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #ef4444;
        color: #f8fafc;
    }
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .result-prob {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    
    /* Dynamic Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sub-section headers */
    .section-header {
        border-left: 5px solid #38bdf8;
        padding-left: 10px;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #38bdf8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Load Assets & Cache
# ----------------------------------------------------
@st.cache_resource
def load_models_and_metadata():
    if not os.path.exists("models/metadata.json"):
        return None, None
        
    with open("models/metadata.json", "r") as f:
        metadata = json.load(f)
        
    assets = {}
    loaders = {
        "num_imp": "num_imp.pkl",
        "cat_imp": "cat_imp.pkl",
        "le_edu": "le_edu.pkl",
        "le_target": "le_target.pkl",
        "ohe": "ohe.pkl",
        "scaler": "scaler.pkl",
        "log_model": "log_model.pkl",
        "knn_model": "knn_model.pkl",
        "nb_model": "nb_model.pkl"
    }
    
    for key, filename in loaders.items():
        with open(f"models/{filename}", "rb") as f:
            assets[key] = pickle.load(f)
            
    return metadata, assets

@st.cache_data
def load_raw_data():
    if os.path.exists("loan_approval_data.csv"):
        return pd.read_csv("loan_approval_data.csv")
    return None

metadata, assets = load_models_and_metadata()
df_raw = load_raw_data()

# Check for existence of model assets
if metadata is None or assets is None:
    st.error("Error: Trained model assets or metadata were not found. Please run 'train_and_save.py' first to build and save the model.")
    st.stop()

# ----------------------------------------------------
# Sidebar Controls
# ----------------------------------------------------
st.sidebar.markdown("<h1 style='text-align: left; margin-top: 0; margin-bottom: 10px; font-size: 3rem;'>🏦</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='margin-top:0;'>CreditWise System</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Model selection in sidebar
model_choice = st.sidebar.selectbox(
    "Select Classifier Model",
    ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Naive Bayes"],
    help="Choose which algorithm will make the loan approval prediction."
)

# Map UI name to model key
model_key_map = {
    "Logistic Regression": "log_model",
    "K-Nearest Neighbors (KNN)": "knn_model",
    "Naive Bayes": "nb_model"
}
selected_model_key = model_key_map[model_choice]

# Display training performance metrics
st.sidebar.subheader("Selected Model Metrics")
stats_map = {
    "Logistic Regression": "LogisticRegression",
    "K-Nearest Neighbors (KNN)": "KNN",
    "Naive Bayes": "NaiveBayes"
}
stats_key = stats_map[model_choice]
stats = metadata["model_stats"][stats_key]

st.sidebar.metric("Accuracy", f"{stats['accuracy']*100:.2f}%")
st.sidebar.metric("Precision", f"{stats['precision']*100:.2f}%")
st.sidebar.metric("Recall", f"{stats['recall']*100:.2f}%")
st.sidebar.metric("F1 Score", f"{stats['f1']*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("<div style='font-size:0.8rem; color:#888;'>CreditWise Loan Approval System • Version 1.0.0</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Main Panel Layout
# ----------------------------------------------------
st.markdown("<h1 class='gradient-text' style='margin-bottom: 0;'>💳 CreditWise Intelligent Loan System</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.1rem; color:#555;'>Analyze creditworthiness, predict loan approval, and explore portfolio metrics instantly.</p>", unsafe_allow_html=True)

# Creating Tabs
tab_predict, tab_analytics = st.tabs(["🔮 Loan Approval Predictor", "📊 Data Insights & Analytics"])

# ----------------------------------------------------
# Tab 1: Predictor Dashboard
# ----------------------------------------------------
with tab_predict:
    st.markdown("<h3 class='section-header'>Loan Applicant Profiler</h3>", unsafe_allow_html=True)
    st.write("Fill out the applicant details below to analyze creditworthiness and predict approval status.")
    
    # Pre-population values from metadata
    numerical_metadata = metadata["numerical"]
    categorical_metadata = metadata["categorical"]
    
    # Setting up inputs in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics & Personal Details**")
        age = st.slider(
            "Applicant Age", 
            min_value=int(numerical_metadata["Age"]["min"]), 
            max_value=int(numerical_metadata["Age"]["max"]), 
            value=int(numerical_metadata["Age"]["median"]),
            step=1
        )
        gender = st.selectbox("Gender", categorical_metadata["Gender"])
        marital_status = st.selectbox("Marital Status", categorical_metadata["Marital_Status"])
        dependents = st.selectbox(
            "Number of Dependents", 
            options=[0, 1, 2, 3],
            index=int(numerical_metadata["Dependents"]["median"])
        )
        education_level = st.selectbox("Education Level", categorical_metadata["Education_Level"])
        
    with col2:
        st.markdown("**Financial Standing ($)**")
        app_income = st.number_input(
            "Applicant Annual Income ($)", 
            min_value=float(numerical_metadata["Applicant_Income"]["min"]), 
            max_value=float(numerical_metadata["Applicant_Income"]["max"]), 
            value=float(numerical_metadata["Applicant_Income"]["median"]),
            step=500.0,
            format="%.0f"
        )
        coapp_income = st.number_input(
            "Coapplicant Annual Income ($)", 
            min_value=float(numerical_metadata["Coapplicant_Income"]["min"]), 
            max_value=float(numerical_metadata["Coapplicant_Income"]["max"]), 
            value=float(numerical_metadata["Coapplicant_Income"]["median"]),
            step=250.0,
            format="%.0f"
        )
        savings = st.number_input(
            "Total Savings ($)", 
            min_value=float(numerical_metadata["Savings"]["min"]), 
            max_value=float(numerical_metadata["Savings"]["max"]), 
            value=float(numerical_metadata["Savings"]["median"]),
            step=500.0,
            format="%.0f"
        )
        collateral = st.number_input(
            "Collateral Asset Value ($)", 
            min_value=float(numerical_metadata["Collateral_Value"]["min"]), 
            max_value=float(numerical_metadata["Collateral_Value"]["max"]), 
            value=float(numerical_metadata["Collateral_Value"]["median"]),
            step=1000.0,
            format="%.0f"
        )
        
    with col3:
        st.markdown("**Loan & Credit Details**")
        credit_score = st.slider(
            "Credit Score", 
            min_value=int(numerical_metadata["Credit_Score"]["min"]), 
            max_value=int(numerical_metadata["Credit_Score"]["max"]), 
            value=int(numerical_metadata["Credit_Score"]["median"]),
            help="Higher scores reflect lower credit risk."
        )
        dti_ratio = st.slider(
            "Debt-to-Income (DTI) Ratio", 
            min_value=float(numerical_metadata["DTI_Ratio"]["min"]), 
            max_value=float(numerical_metadata["DTI_Ratio"]["max"]), 
            value=float(numerical_metadata["DTI_Ratio"]["median"]),
            step=0.01,
            help="Your total monthly debt payments divided by your gross monthly income."
        )
        loan_amount = st.number_input(
            "Requested Loan Amount ($)", 
            min_value=float(numerical_metadata["Loan_Amount"]["min"]), 
            max_value=float(numerical_metadata["Loan_Amount"]["max"]), 
            value=float(numerical_metadata["Loan_Amount"]["median"]),
            step=500.0,
            format="%.0f"
        )
        loan_term = st.slider(
            "Loan Term (Months)", 
            min_value=int(numerical_metadata["Loan_Term"]["min"]), 
            max_value=int(numerical_metadata["Loan_Term"]["max"]), 
            value=int(numerical_metadata["Loan_Term"]["median"]),
            step=12
        )
        
    # Additional categoricals expanding in full columns below
    col_purpose, col_property, col_emp_status, col_emp_cat = st.columns(4)
    with col_purpose:
        loan_purpose = st.selectbox("Loan Purpose", categorical_metadata["Loan_Purpose"])
    with col_property:
        property_area = st.selectbox("Property Location Area", categorical_metadata["Property_Area"])
    with col_emp_status:
        emp_status = st.selectbox("Employment Status", categorical_metadata["Employment_Status"])
    with col_emp_cat:
        employer_category = st.selectbox("Employer Industry/Category", categorical_metadata["Employer_Category"])
        
    existing_loans = st.selectbox(
        "Number of Active/Existing Loans",
        options=[0, 1, 2, 3, 4],
        index=int(numerical_metadata["Existing_Loans"]["median"])
    )

    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Evaluate Loan Application", type="primary", use_container_width=True):
        # 1. Construct user input DataFrame
        input_data = {
            "Applicant_Income": app_income,
            "Coapplicant_Income": coapp_income,
            "Employment_Status": emp_status,
            "Age": float(age),
            "Marital_Status": marital_status,
            "Dependents": float(dependents),
            "Credit_Score": float(credit_score),
            "Existing_Loans": float(existing_loans),
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": loan_amount,
            "Loan_Term": float(loan_term),
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category
        }
        
        input_df = pd.DataFrame([input_data])
        
        # 2. Impute (handles potential empty selections, though widgets guarantee complete numbers here)
        numerical_cols_list = list(numerical_metadata.keys())
        categorical_cols_list = list(categorical_metadata.keys())
        # Reorder to match fit alignment
        input_df[numerical_cols_list] = assets["num_imp"].transform(input_df[numerical_cols_list])
        
        # 3. Encode Education Level
        input_df["Education_Level"] = assets["le_edu"].transform(input_df["Education_Level"])
        
        # 4. One Hot Encode remaining categorical columns
        ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
        encoded = assets["ohe"].transform(input_df[ohe_cols])
        encoded_df = pd.DataFrame(encoded, columns=assets["ohe"].get_feature_names_out(ohe_cols), index=input_df.index)
        
        # Concat and Drop original categorical cols
        input_df = pd.concat([input_df.drop(columns=ohe_cols), encoded_df], axis=1)
        
        # 5. Feature Engineering: Squared features as trained
        input_df["DTI_Ratio_sq"] = input_df["DTI_Ratio"] ** 2
        input_df["Credit_Score_sq"] = input_df["Credit_Score"] ** 2
        
        # Drop linear Credit Score and DTI ratio
        input_df = input_df.drop(columns=["DTI_Ratio", "Credit_Score"])
        
        # Reorder features exactly to match training structure
        feature_names = metadata["feature_names"]
        input_df = input_df[feature_names]
        
        # 6. Scaling
        input_scaled = assets["scaler"].transform(input_df)
        
        # 7. Predict & Compute Probabilities
        model = assets[selected_model_key]
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        approval_probability = probabilities[1]
        
        # 8. Display Results Page
        st.markdown("<h3 class='section-header'>Risk Assessment Output</h3>", unsafe_allow_html=True)
        
        result_col1, result_col2 = st.columns([2, 1])
        
        with result_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card approved-card">
                    <div class="result-title">🎉 Loan Application Approved</div>
                    <div class="result-prob">The system predicted approval with a <strong>{approval_probability*100:.2f}%</strong> confidence score.</div>
                    <div style="font-size:1rem; line-height: 1.6;">
                        <strong>Key Strengths:</strong><br/>
                        <ul>
                            <li>Strong credit history score of {credit_score:.0f}.</li>
                            <li>Balanced Debt-to-Income ratio of {dti_ratio:.2f}.</li>
                            <li>Collateral assets cover the requested loan amount of ${loan_amount:,.2f}.</li>
                        </ul>
                        The applicant meets the safety threshold limits for credit distribution.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card rejected-card">
                    <div class="result-title">❌ Loan Application Denied</div>
                    <div class="result-prob">The system predicted denial with a <strong>{(1-approval_probability)*100:.2f}%</strong> risk probability.</div>
                    <div style="font-size:1rem; line-height: 1.6;">
                        <strong>Potential Risk Indicators:</strong><br/>
                        <ul>
                            <li>A Credit Score of {credit_score:.0f} is below target benchmarks.</li>
                            <li>The DTI Ratio ({dti_ratio:.2f}) represents high leverage relative to income.</li>
                            <li>Savings account balance is low relative to the request.</li>
                        </ul>
                        We recommend building up savings, paying down other debts, or applying with a higher collateral cover.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with result_col2:
            # Gauge Indicator of approval probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = approval_probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Approval Probability (%)", 'font': {'size': 18, 'color': '#f8fafc'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8", 'tickfont': {'color': '#f8fafc'}},
                    'bar': {'color': "#38bdf8"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [40, 70], 'color': 'rgba(234, 179, 8, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                height=250,
                font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------
# Tab 2: Analytics & Insights Dashboard
# ----------------------------------------------------
with tab_analytics:
    if df_raw is not None:
        st.markdown("<h3 class='section-header'>Overall Portfolio Metrics</h3>", unsafe_allow_html=True)
        
        # Calculate KPI variables
        total_records = len(df_raw)
        
        # Check target distribution
        raw_approvals = df_raw["Loan_Approved"].dropna().value_counts()
        yes_count = raw_approvals.get("Yes", 0)
        no_count = raw_approvals.get("No", 0)
        total_labeled = yes_count + no_count
        approval_rate = (yes_count / total_labeled) * 100 if total_labeled > 0 else 0.0
        
        avg_income = df_raw["Applicant_Income"].mean()
        avg_credit = df_raw["Credit_Score"].mean()
        
        # Display KPI Grid
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Total Applicants</div>
                <div class="kpi-val">{total_records:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Historical Approval Rate</div>
                <div class="kpi-val">{approval_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Avg Applicant Income</div>
                <div class="kpi-val">${avg_income:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi_col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-lbl">Average Credit Score</div>
                <div class="kpi-val">{avg_credit:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<h3 class='section-header'>Visual Analytics</h3>", unsafe_allow_html=True)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # 1. Credit Score distribution vs approval status
            fig_credit = px.histogram(
                df_raw.dropna(subset=["Loan_Approved"]), 
                x="Credit_Score", 
                color="Loan_Approved",
                barmode="overlay",
                color_discrete_map={"Yes": "#22c55e", "No": "#ef4444"},
                title="Credit Score Distribution by Loan Approval Status",
                opacity=0.7,
                labels={"Credit_Score": "Credit Score", "Loan_Approved": "Approved"}
            )
            fig_credit.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_credit, use_container_width=True)
            
        with chart_col2:
            # 2. Debt-to-Income vs Approval Boxplot
            fig_dti = px.box(
                df_raw.dropna(subset=["Loan_Approved"]), 
                x="Loan_Approved", 
                y="DTI_Ratio",
                color="Loan_Approved",
                color_discrete_map={"Yes": "#22c55e", "No": "#ef4444"},
                title="Debt-to-Income (DTI) Ratio by Approval Status",
                labels={"Loan_Approved": "Loan Approved", "DTI_Ratio": "DTI Ratio"}
            )
            fig_dti.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_dti, use_container_width=True)
            
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            # 3. Loan Amount vs income scatter plot
            fig_scatter = px.scatter(
                df_raw.dropna(subset=["Loan_Approved"]), 
                x="Applicant_Income", 
                y="Loan_Amount", 
                color="Loan_Approved",
                color_discrete_map={"Yes": "#22c55e", "No": "#ef4444"},
                opacity=0.6,
                title="Loan Amount vs. Applicant Income",
                labels={"Applicant_Income": "Applicant Income ($)", "Loan_Amount": "Loan Amount ($)", "Loan_Approved": "Approved"}
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with chart_col4:
            # 4. Correlation Heatmap
            corr_df = df_raw.select_dtypes(include="number").drop(columns=["Applicant_ID"], errors="ignore")
            # Fill na with mean just for correlation calculation
            corr_df = corr_df.fillna(corr_df.mean())
            corr_matrix = corr_df.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Numeric Feature Correlation Heatmap"
            )
            fig_corr.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
    else:
        st.warning("Data file 'loan_approval_data.csv' not found. Analytics dashboard features are disabled.")
