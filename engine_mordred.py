from rdkit import Chem
from tqdm import tqdm
import os
import subprocess
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from mordred import Calculator, descriptors
import pandas as pd

class MordredDescriptors:
    def __init__(self, data):
        self.data = data
    
    def conformer_generation(self, xyz_dir, mol_dir):
        smi = self.data['SMILES']
        mol = [Chem.MolFromSmiles(x) for x in smi]
        mol_H = [Chem.AddHs(x) for x in mol]
        for idx, m in tqdm(enumerate(mol_H)):
            if m.GetNumAtoms() > 90:
                rdDistGeom.EmbedMolecule(m, maxAttempts=2000, randomSeed=0)
            else:
                rdDistGeom.EmbedMolecule(m, randomSeed=0)
        for idx, m in enumerate(mol_H):
            Chem.rdmolfiles.MolToXYZFile(m, os.path.join(xyz_dir, f'mol_{idx}.xyz'))
        for idx, m in enumerate(mol_H):
            Chem.rdmolfiles.MolToMolFile(m, os.path.join(mol_dir, f'mol_{idx}.mol'))
    
    def conformer_optimization(self, input_dir, out_dir):
        for file_name in os.listdir(input_dir):
            idx = file_name.split('_')[1].split('.')[0]
            input_file = os.path.join(input_dir, file_name)
            command = f'xtb {input_file} --opt tight --alpb water'
            subprocess.run(command, shell=True)
            file_to_check = 'xtbopt.xyz'
            if os.path.exists(file_to_check):
                xtbopt_path = file_to_check
                new_file_path = os.path.join(out_dir, f'xtbopt_{idx}.xyz')
                os.rename(xtbopt_path, new_file_path)
    
    def xyz_to_mol(self, input_dir, mol_ref_dir, mol_out_dir):
        for idx in range(len(self.data)):
            atoms = []
            xyz_optimized_coords = []
            xyz_file = os.path.join(input_dir, f'xtbopt_{idx}.xyz')
            if os.path.exists(xyz_file):
                with open(xyz_file, 'r') as f:
                    for line_number,line in enumerate(f):
                        if line_number == 0:
                            num_atoms = int(line)
                        elif line_number == 1:
                            comment = line
                        else:
                            atomic_symbol, x, y, z = line.split()
                            atoms.append(atomic_symbol)
                            xyz_optimized_coords.append([float(x),float(y),float(z)])
                m = Chem.MolFromMolFile(os.path.join(mol_ref_dir, f'mol_{idx}.mol'), removeHs=False)
                conf = m.GetConformer()
                for i in range(m.GetNumAtoms()):
                   x,y,z = xyz_optimized_coords[i]
                   conf.SetAtomPosition(i,Point3D(x,y,z))
                Chem.MolToMolFile(m, os.path.join(mol_out_dir, f'xtbopt_{idx}.mol'))
    
    def mordred_descs(self):
        self.conformer_generation(xyz_dir='./data/3D_conformers/rdkit/xyz', mol_dir='./data/3D_conformers/rdkit/mol')
        self.conformer_optimization(input_dir='./data/3D_conformers/rdkit/xyz', out_dir='./data/3D_conformers/xtb/xyz')
        self.xyz_to_mol(input_dir='./data/3D_conformers/xtb/xyz',
                        mol_ref_dir='./data/3D_conformers/rdkit/mol',
                        mol_out_dir='./data/3D_conformers/xtb/mol')
        calc = Calculator(descriptors, ignore_3D=False)
        desc_list = []
        smi_idx = []
        for idx in range(len(self.data)):
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
        smi = self.data['SMILES'][smi_idx]
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

def main():
    data = pd.read_csv('./data/eval_1_combined/dataset/3D_optimizable_dataset.csv')
    calc = MordredDescriptors(data)
    calc.mordred_descs()

if __name__ == "__main__":
    main()