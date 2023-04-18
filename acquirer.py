import metrics
import numpy as np
import math
import heapq
from tqdm import tqdm
from timeit import default_timer

class Acquirer:
    def __init__(self, size, init_size=0.01, metric='random',
                threshold=None, beta=None,
                xi=None, stochastic_preds=None, verbose=0,
                epsilon=0.0,
                ):
        self.size = size
        self.init_size = init_size*self.size
        self.metric = metric

        self.threshold = threshold
        self.beta = beta
        self.xi = xi
        self.stochastic_preds = stochastic_preds

        self.verbose = verbose
        self.epsilon = epsilon

    def _len_(self):
        return self.size
    
    @property
    def needs(self):
        return metrics.get_needs(self.metric)
    
    @property
    def batch_sizes(self):
        return self.__batch_sizes

    @batch_sizes.setter
    def batch_sizes(self, batch_sizes):
        self.__batch_sizes = [bs for bs in batch_sizes]

        for i, bs in enumerate(self.__batch_sizes):
            if isinstance(bs, float):
                if bs < 0 or bs > 1:
                    raise ValueError(f"batch_size(={bs} must be in [0, 1]")
                self.__batch_sizes[i] = math.ceil(self.size * bs)
            if bs < 0:
                raise ValueError(f"batch_size(={bs} must be positive")

    def batch_size(self, t):
        try:
            batch_size = self.batch_sizes[t]
        except (IndexError, TypeError):
            batch_size = self.batch_sizes[-1]

        return batch_size

    def acquire_initial(self, xs):
        U = metrics.random(np.empty(self.size))

        heap = []
        for x, u in tqdm(zip(xs, U), total=U.size, desc="Acquiring"):
            if len(heap) < self.init_size:
                heapq.heappush(heap, (u, x))
            else:
                heapq.heappushpop(heap, (u, x))
        
        return [x for _, x in heap]

    def acquire_batch(
        self,xs,y_means,y_vars,
        explored = None,k = 1,
        cluster_ids = None,
        cluster_sizes = None,
        t = None,**kwargs,
    ):
        if explored:
            ys = list(explored.values())
            Y = np.nan_to_num(np.array(ys, float), nan=-np.inf)
            current_max = np.partition(Y, -k)[-k] if len(Y) >= k else Y.max()
        else:
            explored = {}
            current_max = float("-inf")

        batch_size = self.init_size

        begin = default_timer()

        Y_mean = np.array(y_means)
        Y_var = np.array(y_vars)

        if self.verbose > 1:
            print("Calculating acquisition utilities ...", end=" ")

        U = metrics.calc(
            self.metric,
            Y_mean,
            Y_var,
            current_max,
            self.threshold,
            self.beta,
            self.xi,
            self.stochastic_preds,
        )

        # idxs = np.random.choice(U.size, math.ceil(batch_size * self.epsilon), False)
        # U[idxs] = np.inf

        if self.verbose > 1:
            print("Done!")
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f"      Utility calculation took {mins}m {secs}s")

        if cluster_ids is None and cluster_sizes is None:
            heap = []
            for x, u in tqdm(zip(xs, U), "Acquiring", U.size, disable=self.verbose < 1):
                if x in explored:
                    continue

                if len(heap) < batch_size:
                    heapq.heappush(heap, (u, x))
                else:
                    heapq.heappushpop(heap, (u, x))
        else:
            raise NotImplementedError
        if self.verbose > 1:
            print(f"Selected {len(heap)} new samples")
        if self.verbose > 2:
            total = default_timer() - begin
            mins, secs = divmod(int(total), 60)
            print(f"      Batch acquisition took {mins}m {secs}s")

        return [x for _, x in sorted(heap, reverse=True)]