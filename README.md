# SMILES-based deep learning model combined with active learning accelerating conventional molecular docking procedure
This is the code for "SMILES-based deep learning model combined with active learning accelerating conventional molecular docking procedure" paper. 


# Conda Environemt Setup
conda env create -f environment.yaml


# Dataset
The database with docking scores for virtual screening are provided in ```data``` folder. The fingerprints or the whole database of compounds are provided in ```libraries```.


# Quickstart
# Step 1: Randomly select compounds
Run the following command in the terminal
```python run.py --iteration 0 --path="output/Enamine10k" --model_type="smiles" --libraries="libraries/Enamine10k.csv.gz" --score_path="data/Enamine10k_scores.csv.gz" --uncertainty="dropout" --metric="greedy" --init_size=0.01 --explore_size=0.01```
or using fingerprints
```python run.py --iteration 0 --path="output/Enamine10k" --model_type="fingerprint" --fingerprint="pair" --fps_path="feature/libraries/Enamine10k.h5" --libraries="libraries/Enamine10k.csv.gz" --score_path="data/Enamine10k_scores.csv.gz" --uncertainty="dropout" --metric="greedy" --init_size=0.01 --explore_size=0.01```

# Step 2. Train the model 
Run the following command in the terminal
```python run.py --iteration 1 --path="output/Enamine10k" --model_type="smiles" --libraries="libraries/Enamine10k.csv.gz" --score_path="data/Enamine10k_scores.csv.gz" --uncertainty="dropout" --metric="greedy" --init_size=0.01 --explore_size=0.01```
or using fingerprints
```python run.py --iteration 1 --path="output/Enamine10k" --model_type="fingerprint" --fingerprint="pair" --fps_path="libraries/Enamine10k.h5" --libraries="libraries/Enamine10k.csv.gz" --score_path="data/Enamine10k_scores.csv.gz" --uncertainty="dropout" --metric="greedy" --init_size=0.01 --explore_size=0.01```