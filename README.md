# Exploring the Effectiveness of Three-Dimensional Molecular Representations in Caco-2 Permeability Prediction
This repository contains the necessary scripts to replicate the work performed.

**1. main folder**

Contains the main scripts for evaluations.

<br>

**2. generate_conformers.py & calc_mordred.py**

Used to generate 3D conformers for mordred 3D descriptor calculations, and calculate the corresponding mordred descriptors

<br>

**3. embeddings.py, engine.py, visualization.py**

Contains helper functions to extract embeddings, assist in model training and testing, and plotting.

<br>

To replicate this project, follow the steps below depending on whether you want to recalculate descriptors or use the precomputed ones. Please ensure all necessary directories are in place, which can be found in the link, before executing the scripts. All trained models and data files can be retrieved from [here](blank).
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
1. Download the precomputed Mordred descriptors.
2. Set up the main environment:
```bash
conda env create -f environment.yml --name Caco2-3D
conda activate Caco2-3D
```
3. Run main code.
