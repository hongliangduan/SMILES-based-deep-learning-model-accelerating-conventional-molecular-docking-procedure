import csv
from functools import partial
import gzip
from pathlib import Path
from typing import Dict, Iterable, Optional

from configargparse import ArgumentParser
from tqdm import tqdm

from objectives.base import Objective


class LookupObjective(Objective):
    """A LookupObjective calculates the objective function by looking the
    value up in an input file.

    Useful for retrospective studies.

    Attributes
    ----------
    self.data : Dict[str, Optional[float]]
        a dictionary containing the objective function value of each molecule

    Parameters
    ----------
    objective_config : str
        the configuration file for a LookupObjective
    **kwargs
        unused and addditional keyword arguments
    """

    def __init__(self, minimize, path, **kwargs):
        path = path
        delimiter = ','
        title_line = True
        smiles_col = 0
        score_col = 1

        if Path(path).suffix == ".gz":
            open_ = partial(gzip.open, mode="rt")
        else:
            open_ = open

        self.data = {}
        with open_(path) as fid:
            reader = csv.reader(fid, delimiter=delimiter)
            if title_line:
                next(fid)

            for row in tqdm(reader, "Building oracle", leave=False):
                key = row[smiles_col]
                val = row[score_col]
                try:
                    self.data[key] = float(val)
                except ValueError:
                    pass

        super().__init__(minimize=minimize)

    def forward(self, smis: Iterable[str], *args, **kwargs) -> Dict[str, Optional[float]]:
        return {smi: self.c * self.data[smi] if smi in self.data else None for smi in smis}