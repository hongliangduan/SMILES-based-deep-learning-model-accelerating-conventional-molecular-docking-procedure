from dataclasses import dataclass
from itertools import chain, islice
import math
import numpy as np
import ray

import rdkit
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as rdmd
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

try:
    from map4 import map4
except ImportError:
    pass

import psutil as ps

@dataclass
class Featurizer:
    fingerprint: str = "pair"
    radius: int = 2
    length: int = 2048

    def __post_init__(self):
        if self.fingerprint == "maccs":
            self.radius = 0
            self.length = 167

    def __len__(self):
        return self.length

    def __call__(self, smi: str):
        return featurize(smi, self.fingerprint, self.radius, self.length)


def featurize(smi, fingerprint, radius, length):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None

    if fingerprint == 'morgan':
        fps = rdmd.GetMorganFingerprintAsBitVect(mol, radius=radius, useChirality=True)
    elif fingerprint == "pair":
        fps = rdmd.GetHashedAtomPairFingerprintAsBitVect(
            mol, minLength=1, maxLength=1 + radius, nBits=length
        )
    elif fingerprint == "rdkit":
        fps = Chem.RDKFingerprint(mol, minPath=1, maxPath=1 + radius, fpSize=length)
    elif fingerprint == "maccs":
        fps = rdmd.GetMACCSKeysFingerprint(mol)
    elif fingerprint == "map4":
        fps = map4.MAP4Calculator(dimensions=length, radius=radius, is_folded=True).calculate(mol)
    else:
        return NotImplementedError(f'Unrecognized fingerprint:"{fingerprint}".')
    
    X = np.empty(len(fps))
    ConvertToNumpyArray(fps, X)
    return X

@ray.remote
def featurize_batch(smis, fingerprint, radius, length):
    return [featurize(smi, fingerprint, radius, length) for smi in smis]

def feature_matrix(smis, featurizer, disable: bool = False):
    fingerprint = featurizer.fingerprint
    radius = featurizer.radius
    length = len(featurizer)

    chunksize = int(math.sqrt(ps.cpu_count()) * 1024)
    refs = [
        featurize_batch.remote(smis, fingerprint, radius, length)
        for smis in batches(smis, chunksize)
    ]
    fps_chunks = [
        ray.get(r) for r in tqdm(refs, "Featurizing", leave=False, disable=disable, unit="smi")
    ]

    return list(chain(*fps_chunks))

def batches(it, size: int):
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])