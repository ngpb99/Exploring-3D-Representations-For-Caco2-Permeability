from mordred import Calculator, descriptors
import pandas as pd
from rdkit import Chem
import os

def mordred_descs(data):
    calc = Calculator(descriptors, ignore_3D=False)
    desc_list = []
    smi_idx = []
    for idx in range(len(data)):
        mol_file = f'./data/3D_conformers/xtb/mol/xtbopt_{idx}.mol'
        if os.path.exists(mol_file):
            mol = Chem.MolFromMolFile(mol_file, removeHs=False)
            desc = calc(mol)
            desc_list.append(desc)
            smi_idx.append(idx)
    df_desc = pd.DataFrame(desc_list, columns=[str(key) for key in desc.keys()])
    var_per_col = list(df_desc.var())
    zero_var_desc = []
    for idx, var in enumerate(var_per_col):
        if var == 0:
            zero_var_desc.append(idx)
    df_desc = df_desc.drop(columns=df_desc.columns[zero_var_desc])
    smi = data['SMILES'][smi_idx]
    smi.reset_index(drop=True, inplace=True)
    df_desc['SMILES'] = smi
    # cleaning NaN in descriptors
    obj_cols = df_desc.select_dtypes(include=['object']).columns
    unchanged_cols = []
    for col in obj_cols:
        try:
            df_desc[col] = df_desc[col].astype('float')
        except ValueError:
            unchanged_cols.append(col)
    col_na = df_desc.isna().sum(axis=0)
    cols_to_drop = col_na[col_na > 50].index # Drop cols with more than 50 NA
    df_desc.drop(columns=cols_to_drop, inplace=True)
    df_desc.dropna(axis=0, inplace=True) # Drop rows with NA
    df_desc.reset_index(inplace=True, drop=True)
    bool_cols = df_desc.select_dtypes(include=['bool']).columns
    df_desc.loc[:, bool_cols] = df_desc.loc[:, bool_cols].astype(int)
    train = pd.read_csv('./data/eval_1_combined/dataset/train_data.csv')
    test = pd.read_csv('./data/eval_1_combined/dataset/test_data.csv')
    train_mordred = pd.merge(train, df_desc, on='SMILES', how='inner')
    test_mordred = pd.merge(test, df_desc, on='SMILES', how='inner')
    train_mordred.to_csv('./data/eval_1_combined/predefined_descs/mordred_train.csv', index=False)
    test_mordred.to_csv('./data/eval_1_combined/predefined_descs/mordred_test.csv', index=False)

if __name__ == "__main__":
    data = pd.read_csv('./data/eval_1_combined/dataset/3D_optimizable_dataset.csv')
    mordred_descs(data)