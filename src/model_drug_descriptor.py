import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# 定义一个函数来计算单个SMILES的分子描述符
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

# 读取数据
data = pd.read_csv(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\smiles.csv')

# SMILES列和后缀定义
smiles_columns = {
    'API_smiles': 'API',
    'Polymer_smiles': 'excip1',
    'Surfant_smiles': 'excip2',
    'Diss_surfant_smiles': 'surf'
}

# 预处理：将'0'替换为数字0
for col in smiles_columns.keys():
    data[col] = data[col].replace('0', 0)

# 计算每列SMILES的描述符并添加到DataFrame中
for col, suffix in smiles_columns.items():
    print(f"Processing {col}")
    descriptors = data[col].apply(lambda x: pd.Series(calculate_descriptors(x, suffix)))
    data = pd.concat([data, descriptors], axis=1)

# 手动指定最终列顺序
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
    "pH_shift", "Output_time", "Output_initial_fraction"
]

# 重新排序列（确保所有列在data中才选）
ordered_columns_final = [col for col in ordered_columns if col in data.columns]
data = data[ordered_columns_final]



# 保存结果
data.to_csv(r'C:\Users\熊萍\Desktop\固体分散体\溶出_ZJY\dissolution\test.csv', index=False)
print("Descriptors calculated and saved.")

# 打印前几行
print(data.head())
