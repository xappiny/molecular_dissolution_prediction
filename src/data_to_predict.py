import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
import joblib
import numpy as np
from pathlib import Path

# Define file paths
RAW_DATA_PATH = r'C:\Users\dissolution\smiles.csv'
PROCESSED_DATA_PATH = r'C:\Users\dissolution\test.csv'
SCALER_PATH = r'C:\Users\dissolution\external_val\scaler.pkl'
TABPFN_MODEL_PATH = r'C:\Users\dissolution\external_val\tabpfn_regressor.pkl'
RF_MODEL_PATH = r'C:\Users\dissolution\external_val\best_rf_fold4_global.pkl'
LGBM_MODEL_PATH = r'C:\Users\dissolution\external_val\best_lgb_fold5_global.pkl'
OUTPUT_PATH = r'C:\Users\dissolution\external_val\external_prediction_molecular.csv'

# Function to calculate molecular descriptors from SMILES
def calculate_descriptors(smiles, suffix):
    if smiles == 0 or smiles == '0':
        return {f"MolWt_{suffix}": 0,
                f"Complexity_{suffix}": 0,
                f"TPSA_{suffix}": 0,
                f"HeavyAtomCount_{suffix}": 0,
                f"NumHAcceptors_{suffix}": 0,
                f"NumHDonors_{suffix}": 0,
                f"NumRotatableBonds_{suffix}": 0,
                f"RingCount_{suffix}": 0,
                f"MolLogP_{suffix}": 0}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {f"MolWt_{suffix}": float('nan'),
                    f"Complexity_{suffix}": float('nan'),
                    f"TPSA_{suffix}": float('nan'),
                    f"HeavyAtomCount_{suffix}": float('nan'),
                    f"NumHAcceptors_{suffix}": float('nan'),
                    f"NumHDonors_{suffix}": float('nan'),
                    f"NumRotatableBonds_{suffix}": float('nan'),
                    f"RingCount_{suffix}": float('nan'),
                    f"MolLogP_{suffix}": float('nan')}

        descriptors = {
            f"MolWt_{suffix}": Descriptors.MolWt(mol),
            f"Complexity_{suffix}": Descriptors.BertzCT(mol),
            f"TPSA_{suffix}": Descriptors.TPSA(mol),
            f"HeavyAtomCount_{suffix}": Descriptors.HeavyAtomCount(mol),
            f"NumHAcceptors_{suffix}": Descriptors.NumHAcceptors(mol),
            f"NumHDonors_{suffix}": Descriptors.NumHDonors(mol),
            f"NumRotatableBonds_{suffix}": Descriptors.NumRotatableBonds(mol),
            f"RingCount_{suffix}": Descriptors.RingCount(mol),
            f"MolLogP_{suffix}": Crippen.MolLogP(mol)
        }
        return descriptors
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return {f"MolWt_{suffix}": float('nan'),
                f"Complexity_{suffix}": float('nan'),
                f"TPSA_{suffix}": float('nan'),
                f"HeavyAtomCount_{suffix}": float('nan'),
                f"NumHAcceptors_{suffix}": float('nan'),
                f"NumHDonors_{suffix}": float('nan'),
                f"NumRotatableBonds_{suffix}": float('nan'),
                f"RingCount_{suffix}": float('nan'),
                f"MolLogP_{suffix}": float('nan')}

# Function to process SMILES data and calculate descriptors
def process_smiles_data():
    try:
        # Read raw data
        data = pd.read_csv(RAW_DATA_PATH)

        # SMILES columns and suffixes
        smiles_columns = {
            'API_smiles': 'API',
            'Polymer_smiles': 'excip1',
            'Surfant_smiles': 'excip2',
            'Diss_surfant_smiles': 'surf'
        }

        # Preprocess: Replace '0' with numeric 0
        for col in smiles_columns.keys():
            data[col] = data[col].replace('0', 0)

        # Calculate descriptors for each SMILES column
        for col, suffix in smiles_columns.items():
            print(f"Processing {col}")
            descriptors = data[col].apply(lambda x: pd.Series(calculate_descriptors(x, suffix)))
            data = pd.concat([data, descriptors], axis=1)

        # Define ordered columns (excluding SMILES columns)
        ordered_columns = [
            "MolWt_API", "Complexity_API", "TPSA_API", "HeavyAtomCount_API", "NumHAcceptors_API", "NumHDonors_API",
    "NumRotatableBonds_API", "RingCount_API", "MolLogP_API",
    "Complexity_excip1", "TPSA_excip1", "HeavyAtomCount_excip1", "NumHAcceptors_excip1", "NumHDonors_excip1",
    "NumRotatableBonds_excip1", "RingCount_excip1", "MolLogP_excip1",
    "MolWt_excip2", "TPSA_excip2", "HeavyAtomCount_excip2", "NumHAcceptors_excip2", "NumHDonors_excip2",
    "NumRotatableBonds_excip2", "RingCount_excip2", "MolLogP_excip2",
    "MolWt_surf", "TPSA_surf", "HeavyAtomCount_surf", "NumHAcceptors_surf", "NumHDonors_surf",
    "NumRotatableBonds_surf", "RingCount_surf", "MolLogP_surf",
    "API_melting_point", "API_pka", "API_molar_volume", "API_surface_tension", "API_Tg",
    "Polymer_MolWt", "Polymer_repeat_units", "Polymer_Tg",
    "Process_tech", "Process_temp1", "Process_temp2", "Process_sols_ratio",
    "Drug_ratio", "Polymer_ratio", "Surfactant_ratio", "Drug_dose",
    "Diss_volume_donor", "Diss_rotation_speed", "Diss_temp", "Diss_filter_size", "Diss_method", "Diss_pH",
    "pH_shift", "Output_time", "Output_initial_fraction"]

        # Ensure all ordered columns exist in the data
        ordered_columns_final = [col for col in ordered_columns if col in data.columns]
        data = data[ordered_columns_final]

        # Save processed data
        data.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Descriptors calculated and saved to '{PROCESSED_DATA_PATH}'")
        return data
    except Exception as e:
        print(f"Error processing SMILES data: {e}")
        raise

# Function to load models and processed data
def load_data_and_models():
    try:
        # Load pre-trained scaler
        scaler = joblib.load(SCALER_PATH)
        # Load pre-trained models
        tabpfn_model = joblib.load(TABPFN_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        lgbm_model = joblib.load(LGBM_MODEL_PATH)
        # Load processed dataset
        data_new = pd.read_csv(PROCESSED_DATA_PATH, encoding='unicode_escape')
        return scaler, tabpfn_model, rf_model, lgbm_model, data_new
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        raise
    except Exception as e:
        print(f"Error loading data or models: {e}")
        raise

# Function to preprocess data for prediction
def preprocess_data(data_new, scaler):
    try:
        # Select features for prediction (adjust column range as needed)
        X_new = data_new.iloc[:, :]  # Exclude the last column if it's a target or non-feature
        if X_new.empty:
            raise ValueError("No features found in dataset (check column selection)")

        # Standardize features using pre-trained scaler
        X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
        return X_new_scaled, data_new
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

# Function to make predictions
def make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model):
    try:
        predictions = {
            'TabPFN': tabpfn_model.predict(X_new_scaled),
            'RF': rf_model.predict(X_new_scaled),
            'LGBM': lgbm_model.predict(X_new_scaled)
        }
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

# Main execution
def main():
    # Step 1: Process SMILES data to calculate descriptors
    data = process_smiles_data()

    # Step 2: Load models and processed data
    scaler, tabpfn_model, rf_model, lgbm_model, data_new = load_data_and_models()

    # Step 3: Preprocess data for prediction
    X_new_scaled, data_new = preprocess_data(data_new, scaler)

    # Step 4: Make predictions
    predictions = make_predictions(X_new_scaled, tabpfn_model, rf_model, lgbm_model)

    # Step 5: Add predictions to original data
    data_new['Predicted_diss_TabPFN'] = predictions['TabPFN']
    data_new['Predicted_diss_RF'] = predictions['RF']
    data_new['Predicted_diss_LGBM'] = predictions['LGBM']

    # Step 6: Organize predictions by formulation
    has_output_time = 'Output_time' in data_new.columns
    all_predictions = {
        'TabPFN': data_new['Predicted_diss_TabPFN'].values,
        'RF': data_new['Predicted_diss_RF'].values,
        'LGBM': data_new['Predicted_diss_LGBM'].values
    }

    # 步骤 7：打印预测结果
    print("=== Predicted Dissolution Profiles for ISFG Formulations ===")
    for idx in range(len(data_new)):
        print(f"Sample {idx + 1}:")
        for model_name, model_predictions in all_predictions.items():
            print(f"  Model: {model_name}")
            if has_output_time:
                time_point = data_new['Output_time'].iloc[idx]
                diss = model_predictions[idx]
                print(f"    Time: {time_point:.1f} hr, Predicted Dissolution: {diss:.3f}%")
            else:
                diss = model_predictions[idx]
                print(f"    Predicted Dissolution: {diss:.3f}%")
        print("-" * 50)

    # 步骤 8：保存预测结果到 CSV
    try:
        predictions_df = pd.DataFrame({
            **({'time_point': data_new['Output_time']} if has_output_time else {}),
            'Predicted_diss_TabPFN': predictions['TabPFN'],
            'Predicted_diss_RF': predictions['RF'],
            'Predicted_diss_LGBM': predictions['LGBM']
        })
        predictions_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Predictions saved to '{OUTPUT_PATH}'")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise

if __name__ == "__main__":
    main()