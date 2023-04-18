from typing import Callable, Iterable, List, NoReturn, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
from functools import partial
from itertools import islice
from tqdm import tqdm

import gzip
import csv
import h5py
import numpy as np
import pandas as pd

from featurizer import Featurizer, feature_matrix
from model import FinLSTM, SmiLSTM
from acquirer import Acquirer
import objectives

import argparse
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import re
import time

def get_argments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--iteration", default="1" ,help="the number of iteration")
    parser.add_argument(
        "--libraries", 
        default="feature/libraries/Enamine10k.csv.gz",
        help="the path of the molecular pool"
    )
    parser.add_argument(
        "--path", 
        default="feature/output/Enamine10k_(pair,mve,ucb,100,100,0.01,100)/1",
        help="path to save the output csv files",
    )
    parser.add_argument(
        "--fps_path",
        default="feature/libraries/Enamine10k.h5",
        help="the path of the fingerprints of molecular pool"
    )
    parser.add_argument(
        "--score_path",
        default="feature/data/Enamine10k_scores.csv.gz",
        help="the path of the scores of molecular pool"
    )
    parser.add_argument(
        '--metric', 
        default='ucb', 
        choices={"random", "greedy", "threshold", "ts", "ucb", "ei", "pi", "thompson"},
        help="the acquisition metric to use",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        default=True,
        help="whether to minimize the objective function",
    )
    parser.add_argument(
        "--objective",
        default="lookup",
        help="the objective function to use",
    )
    parser.add_argument(
        "--fingerprint",
        default="pair",
        choices={"morgan", "rdkit", "pair", "maccs", "map4"},
        help="the type of fingerprint to use",
    )
    parser.add_argument(
        "--radius", type=int, default=2, help="the radius or path length to use for fingerprints"
    )
    parser.add_argument("--model_type", type=str, default="fingerprint", help="the type of input data")
    parser.add_argument("--length", type=int, default=2048, help="the length of the fingerprint")
    parser.add_argument("--uncertainty", type=str, default='mve')
    parser.add_argument("--batch_size", type=int, default=4096, help="the number of batches to test")
    parser.add_argument("--explore_size", type=float, default=0.01, help="the size of the exploration in the molecular pool")
    parser.add_argument("--dropout_size", type=int, default=10)
    parser.add_argument("--xi", type=float, default=0.01, help="the xi value to use in EI and PI metrics")
    parser.add_argument("--beta", type=int, default=2, help="the beta value to use in the UCB metric")
    parser.add_argument("--threshold",type=float,default=9.0, help="the threshold value as a positive number to use in the threshold metric")
    parser.add_argument("--init_size", type=float, default=0.01)

    return parser.parse_args()


def read_libary(library, delimiter=',', title_line=True, smiles_col=0):
    if Path(library).suffix == ".gz":
        open_ = partial(gzip.open, mode="rt")
    else:
        open_ = open

    with open_(library) as fid:
        reader = csv.reader(fid, delimiter=delimiter)
        if title_line:
            next(reader)

        for row in reader:
            yield row[smiles_col]

def smi_tokenizer(smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

def smis(smis_, libraries):
    if smis_:
        for smi in smis_:
            yield smi
    else:
        for library in libraries:
            for smi in read_libary(library):
                yield smi

def read_finger(path):
    with h5py.File(path, "r") as h5f:
        fps = h5f["fps"]
        for fp in fps:
            yield fp

def batches(it, size):
    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])

def get_predss(
    uncertainty: Optional[str] = None, 
    model = None, 
    xs: Optional[list] = None, 
    dropout_size: Optional[int] = None):

    if uncertainty == 'dropout':
        predss = np.zeros((dropout_size, len(xs)))
        for j in tqdm(
            range(dropout_size), leave=False, desc="bootstrap prediction", unit="pass"
        ):
            predss[j] = model.predict(xs)[:, 0]
        return predss
    elif uncertainty == 'mve':
        return np.log1p(np.exp(-np.abs(xs))) + np.maximum(xs, 0)

def get_means(
    uncertainty: Optional[str] = None, 
    model = None, 
    xs: Optional[list] = None, 
    dropout_size: Optional[int] = None):

    if uncertainty == 'none':
        return model.predict(xs)[:, 0]
    elif uncertainty == 'dropout':
        predss = get_predss(uncertainty,model, xs, dropout_size)
        return np.mean(predss, 0)
    elif uncertainty == 'mve':
        predss = model.predict(xs)
        return predss[:, 0]

def get_means_and_vars(uncertainty, model, xs,dropout_size: Optional[int] = None):
    if uncertainty == 'none':
        raise TypeError("this uncertainty can't predict variances!")
    elif uncertainty == 'dropout':
        predss = get_predss(uncertainty, model, xs, dropout_size)
        return np.mean(predss, 0), np.var(predss, 0)
    elif uncertainty == 'mve':
        predss = model.predict(xs)
        return predss[:, 0], get_predss(uncertainty=uncertainty, xs = predss[:, 1])


def main():
    times = time.time()
    t = time.localtime()

    args = get_argments()

    iteration = int(args.iteration)
    scores = {}
    size = len(pd.read_csv(args.libraries))
    k = int(args.explore_size*size)
    path = args.path
    time_file = str(path) + '/time.txt'

    acquirer = Acquirer(size=size, init_size=args.init_size ,metric=args.metric, beta=args.beta, xi=args.xi, threshold=float(args.threshold))
    objective = objectives.objective(objective=args.objective, minimize=args.minimize, path=args.score_path)
    featurizer = Featurizer(args.fingerprint, args.radius, args.length)

    if args.model_type == 'fingerprint':
        model = FinLSTM(input_size=len(featurizer), uncertainty=args.uncertainty)
    elif args.model_type == 'smiles':
        model = SmiLSTM()

    if iteration == 0:
        inputs = acquirer.acquire_initial(xs=smis(None,libraries=[args.libraries]))
        
        new_scores = objective(inputs)
        scores.update(new_scores)

        road = Path(path)
        road.mkdir(parents=True, exist_ok=True)

        with open(f'{path}/{k+1}_iter_{iteration}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["smiles", "score"])
            writer.writerows(scores.items())

        times = time.time() -times

        with open(time_file, 'a') as timefile:
            timefile.write(f'{t.tm_year}年{t.tm_mon}月{t.tm_mday}日, {t.tm_hour}:{t.tm_min}\n')
            timefile.write(f'{"%.2f"% times}\n')
    else:
        with open(f'{path}/{(k+1)*iteration}_iter_{iteration-1}.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(file)
            for row in tqdm(reader, leave=False):
                key = row[0]
                val = row[1]
                try:
                    scores[key] = float(val)
                except ValueError:
                    scores[key] = None

        xs_ys = scores.items()
        xs, ys = zip(*[(x, y) for x, y in xs_ys if y is not None])
        
        if args.model_type == 'fingerprint':
            model.train(xs, np.array(ys), featurizer=featurizer)

            x_fps = read_finger(path=args.fps_path)
            xs_fps = batches(x_fps, args.batch_size)
            n_batches = (size // args.batch_size) +1

            mean_only = "vars" not in acquirer.needs

            meanss = []
            variancess = []
            
            if mean_only:
                for batch_xs in tqdm(xs_fps, "Inference", total=n_batches, smoothing=0.0, unit="smi"):
                    means = get_means(args.uncertainty, model, batch_xs, args.dropout_size)
                    meanss.append(means)
                    variancess.append([])
            else:
                for batch_xs in tqdm(xs_fps, "Inference", total=n_batches, smoothing=0.0, unit="smi"):
                    means, variances = get_means_and_vars(args.uncertainty, model, batch_xs, args.dropout_size)
                    meanss.append(means)
                    variancess.append(variances)

            meanss = np.concatenate(meanss)
            variancess = np.concatenate(variancess)

        elif args.model_type == 'smiles':
            xss = []
            for x in xs:
                xss.append(smi_tokenizer(x))
            model.get_model(xss)
            model.train(np.array(ys))

            meanss = model.predict(xs=smis(None,libraries=[args.libraries]))
            variancess = []

        inputs = acquirer.acquire_batch(
            xs = smis(None,libraries=[args.libraries]),
            y_means = meanss,
            y_vars = variancess,
            explored = scores,
            k = k,
        )

        new_scores = objective(inputs)
        scores.update(new_scores)

        with open(f'{path}/{(k+1)*(iteration+1)}_iter_{iteration}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["smiles", "score"])
            writer.writerows(scores.items())
        
        times = time.time() -times

        with open(time_file, 'a') as timefile:
            timefile.write(f'{"%.2f"% times}\n')

if __name__ == "__main__":
    main()