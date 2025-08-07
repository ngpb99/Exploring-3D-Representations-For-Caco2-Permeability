# Exploring-3D-Representations-For-Caco2-Permeability
This repository contains the necessary scripts to replicate the work performed.
1. 
To run this project, follow the steps below depending on whether you want to recalculate descriptors or use the precomputed ones.

## Option 1: Recalculate Mordred Descriptors
1. Set up the main environment (used for conformer generation and running all code except Mordred):
```bash
conda env create -f environment.yml --name Caco2-3D
conda activate Caco2-3D
```
2. Run conformer generation (required before Mordred descriptor calculation):
```bash
python generate_conformers.py
```
3. Switch to Mordred environment to calculate descriptors:
```bash
conda env create -f mordred_environment.yml --name mordred
conda activate mordred
python calc_mordred.py
```
4. Switch back to main environment to continue with the rest of the pipeline.

## Option 2: Use Precomputed Descriptors (Recommended)
1. Download the precomputed Mordred descriptors here.
2. Set up the main environment:
```bash
conda env create -f environment.yml --name Caco2-3D
conda activate Caco2-3D
```
3. Run main code.
