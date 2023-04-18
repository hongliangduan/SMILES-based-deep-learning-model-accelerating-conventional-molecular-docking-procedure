from typing import Callable, Optional, Set

import numpy as np
from scipy.stats import norm

RG = np.random.default_rng()


def set_seed(seed: Optional[int] = None):
    global RG
    RG = np.random.default_rng(seed)


def get_metric(metric: str):
    try:
        return {
            "random": random,
            "threshold": threshold,
            "greedy": greedy,
            "noisy": noisy,
            "ucb": ucb,
            "lcb": lcb,
            "thompson": thompson,
            "ts": thompson,
            "ei": ei,
            "pi": pi,
        }[metric]
    except KeyError:
        raise ValueError(f'Unrecognized metric: "{metric}"')


def get_needs(metric: str):
    return {
        "random": set(),
        "greedy": {"means"},
        "noisy": {"means"},
        "ucb": {"means", "vars"},
        "ei": {"means", "vars"},
        "pi": {"means", "vars"},
        "thompson": {"means", "vars"},
        "ts": {"means", "vars"},
        "threshold": {"means"},
    }.get(metric, set())


def valid_metrics() -> Set[str]:
    return {"random", "threshold", "greedy", "noisy", "ucb", "lcb", "ts", "thompson", "ei", "pi"}


def calc(
    metric: str,
    Y_mean: np.ndarray,
    Y_var: np.ndarray,
    current_max: float,
    t: float,
    beta: int,
    xi: float,
    stochastic: bool,
):
    if metric == "random":
        return random(Y_mean)
    if metric == "threshold":
        return threshold(Y_mean, t)
    if metric == "greedy":
        return greedy(Y_mean)
    if metric == "noisy":
        return noisy(Y_mean)
    if metric == "ucb":
        return ucb(Y_mean, Y_var, beta)
    if metric == "lcb":
        return lcb(Y_mean, Y_var, beta)
    if metric in ["ts", "thompson"]:
        return thompson(Y_mean, Y_var, stochastic)
    if metric == "ei":
        return ei(Y_mean, Y_var, current_max, xi)
    if metric == "pi":
        return pi(Y_mean, Y_var, current_max, xi)

    raise ValueError(f'Unrecognized metric "{metric}". Expected one of {valid_metrics()}')


def random(Y_mean: np.ndarray):
    return RG.random(len(Y_mean))


def threshold(Y_mean: np.ndarray, t: float):
    return np.where(Y_mean >= t, RG.random(Y_mean.shape), -1.0)


def greedy(Y_mean: np.ndarray):
    return Y_mean


def noisy(Y_mean: np.ndarray):
    sd = np.std(Y_mean)
    noise = RG.normal(scale=sd, size=len(Y_mean))
    return Y_mean + noise


def ucb(Y_mean: np.ndarray, Y_var: np.ndarray, beta: int = 2):
    return Y_mean + beta * np.sqrt(Y_var)


def lcb(Y_mean: np.ndarray, Y_var: np.ndarray, beta: int = 2):
    return Y_mean - beta * np.sqrt(Y_var)


def thompson(Y_mean: np.ndarray, Y_var: np.ndarray, stochastic: bool = False):
    if stochastic:
        return Y_mean

    Y_sd = np.sqrt(Y_var)

    return RG.normal(Y_mean, Y_sd)


def ei(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float, xi: float = 0.01):
    I = Y_mean - current_max + xi
    Y_sd = np.sqrt(Y_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = I / Y_sd
    E_imp = I * norm.cdf(Z) + Y_sd * norm.pdf(Z)

    mask = Y_var == 0
    E_imp[mask] = I[mask]

    return E_imp


def pi(Y_mean: np.ndarray, Y_var: np.ndarray, current_max: float, xi: float = 0.01):
    I = Y_mean - current_max + xi
    with np.errstate(divide="ignore"):
        Z = I / np.sqrt(Y_var)
    P_imp = norm.cdf(Z)

    mask = Y_var == 0
    P_imp[mask] = np.where(I > 0, 1, 0)[mask]

    return P_imp
