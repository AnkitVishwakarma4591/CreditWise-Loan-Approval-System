import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    print("Loading data...")
    data_path = 'loan_approval_data.csv'
    df = pd.read_csv(data_path)
    
    # 1. Save original categories and ranges for UI metadata before any cleaning/dropping
    metadata = {
        "categorical": {},
        "numerical": {},
        "target_mapping": {}
    }
    
    # Exclude Applicant_ID from metadata features
    features_df = df.drop(columns=["Applicant_ID"], errors="ignore")
    
    for col in features_df.columns:
        if col == "Loan_Approved":
            continue
        if features_df[col].dtype == 'object':
            # Store sorted unique non-null values
            unique_vals = sorted(list(features_df[col].dropna().unique()))
            metadata["categorical"][col] = unique_vals
        else:
            # Store min, max, mean, and median
            non_null = features_df[col].dropna()
            metadata["numerical"][col] = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "median": float(non_null.median())
            }
            
    print("Pre-processing metadata gathered.")

    # 2. Imputation of Missing Values
    categorical_cols = df.select_dtypes(include="object").columns
    numerical_cols = df.select_dtypes(include="number").columns
    
    # We must treat Applicant_ID carefully.
    # Exclude Applicant_ID from imputations as it will be dropped
    numerical_cols = [c for c in numerical_cols if c != 'Applicant_ID']
    categorical_cols = [c for c in categorical_cols]
    
    print("Imputing numerical features...")
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    
    print("Imputing categorical features...")
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
    
    # 3. Drop Applicant_ID
    if 'Applicant_ID' in df.columns:
        df = df.drop("Applicant_ID", axis=1)
        
    # 4. Label Encode Education_Level and Loan_Approved (target)
    print("Encoding binary and target columns...")
    le_edu = LabelEncoder()
    df['Education_Level'] = le_edu.fit_transform(df['Education_Level'])
    
    le_target = LabelEncoder()
    df['Loan_Approved'] = le_target.fit_transform(df['Loan_Approved'])
    
    # Save the target mapping (No=0, Yes=1 usually)
    for class_label, encoded_val in zip(le_target.classes_, le_target.transform(le_target.classes_)):
        metadata["target_mapping"][int(encoded_val)] = str(class_label)
        
    # Save Education mapping in metadata for validation
    metadata["education_mapping"] = {
        str(cls): int(idx) for idx, cls in enumerate(le_edu.classes_)
    }
    
    # 5. One Hot Encoding for other categorical columns
    ohe_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose', 'Property_Area', 'Gender', 'Employer_Category']
    print(f"One-Hot Encoding columns: {ohe_cols}")
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[ohe_cols])
    
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)
    
    # 6. Square features (DTI_Ratio and Credit_Score) as done in Cell 42
    print("Creating squared features...")
    df['DTI_Ratio_sq'] = df['DTI_Ratio'] ** 2
    df['Credit_Score_sq'] = df['Credit_Score'] ** 2
    
    # Drop linear features of DTI_Ratio and Credit_Score as in Cell 42
    x = df.drop(columns=["DTI_Ratio", "Credit_Score", "Loan_Approved"])
    y = df['Loan_Approved']
    
    # Save feature names and order
    metadata["feature_names"] = list(x.columns)
    
    # 7. Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # 8. Scaling
    print("Scaling features...")
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    
    # Create models folder
    os.makedirs("models", exist_ok=True)
    
    # 9. Train Models & Evaluate
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(x_train_scaled, y_train)
    y_pred_log = log_model.predict(x_test_scaled)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
    
    # KNN
    print("Training KNN (n_neighbors=13)...")
    knn_model = KNeighborsClassifier(n_neighbors=13)
    knn_model.fit(x_train_scaled, y_train)
    y_pred_knn = knn_model.predict(x_test_scaled)
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    
    # Naive Bayes
    print("Training Naive Bayes (GaussianNB)...")
    nb_model = GaussianNB()
    nb_model.fit(x_train_scaled, y_train)
    y_pred_nb = nb_model.predict(x_test_scaled)
    print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
    
    # Save accuracies and basic evaluation stats to metadata
    metadata["model_stats"] = {
        "LogisticRegression": {
            "accuracy": float(accuracy_score(y_test, y_pred_log)),
            "precision": float(precision_score(y_test, y_pred_log)),
            "recall": float(recall_score(y_test, y_pred_log)),
            "f1": float(f1_score(y_test, y_pred_log)),
        },
        "KNN": {
            "accuracy": float(accuracy_score(y_test, y_pred_knn)),
            "precision": float(precision_score(y_test, y_pred_knn)),
            "recall": float(recall_score(y_test, y_pred_knn)),
            "f1": float(f1_score(y_test, y_pred_knn)),
        },
        "NaiveBayes": {
            "accuracy": float(accuracy_score(y_test, y_pred_nb)),
            "precision": float(precision_score(y_test, y_pred_nb)),
            "recall": float(recall_score(y_test, y_pred_nb)),
            "f1": float(f1_score(y_test, y_pred_nb)),
        }
    }
    
    # 10. Save all artifacts
    print("\nSaving artifacts to 'models/' directory...")
    
    # Save preprocessors
    with open("models/num_imp.pkl", "wb") as f:
        pickle.dump(num_imp, f)
    with open("models/cat_imp.pkl", "wb") as f:
        pickle.dump(cat_imp, f)
    with open("models/le_edu.pkl", "wb") as f:
        pickle.dump(le_edu, f)
    with open("models/le_target.pkl", "wb") as f:
        pickle.dump(le_target, f)
    with open("models/ohe.pkl", "wb") as f:
        pickle.dump(ohe, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(sc, f)
        
    # Save models
    with open("models/log_model.pkl", "wb") as f:
        pickle.dump(log_model, f)
    with open("models/knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    with open("models/nb_model.pkl", "wb") as f:
        pickle.dump(nb_model, f)
        
    # Save metadata as JSON
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("All models and pipelines trained and saved successfully.")

if __name__ == "__main__":
    main()
