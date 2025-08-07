from rdkit import Chem
from tqdm import tqdm
import os
import subprocess
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
import pandas as pd

class Molecule3DProcessor:
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
    
    def init_all(self):
        self.conformer_generation(xyz_dir='./data/3D_conformers/rdkit/xyz', mol_dir='./data/3D_conformers/rdkit/mol')
        self.conformer_optimization(input_dir='./data/3D_conformers/rdkit/xyz', out_dir='./data/3D_conformers/xtb/xyz')
        self.xyz_to_mol(input_dir='./data/3D_conformers/xtb/xyz',
                        mol_ref_dir='./data/3D_conformers/rdkit/mol',
                        mol_out_dir='./data/3D_conformers/xtb/mol')
    

def main():
    data = pd.read_csv('./data/eval_1_combined/dataset/3D_optimizable_dataset.csv')
    calc = Molecule3DProcessor(data)
    calc.init_all()

if __name__ == "__main__":
    main()