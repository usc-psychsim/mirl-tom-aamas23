import multiprocessing as mp
import os
import tqdm
from typing import Callable, Optional, List
from joblib import Parallel, delayed, parallel_backend
from .logging import MultiProcessLogger, create_mp_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


class _ProgressParallel(Parallel):
    """
    see: https://stackoverflow.com/a/61900501/16031961
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm.tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_parallel(func: Callable,
                 args: List,
                 processes: Optional[int] = None,
                 use_tqdm: bool = True,
                 mp_logging: bool = False) -> List:
    """
    Run the given function for each of the given arguments in parallel and returns  alist with the results.
    :param func: the function to be executed.
    :param list args: the list of arguments for the function to be processed in parallel. If the function has multiple
    arguments, this should be a list of tuples, the length of each should match the function's arity.
    :param int processes: the number of parallel processes to use.  `-1` means all CPUs are used, `1` means no parallel
    computing, `<-1` means `(n_cpus + 1 + n_jobs)` are used, `None` is equivalent to `n_jobs=1`.
    :param bool use_tqdm: whether to show a progress bar during parallel execution.
    :param  bool mp_logging: whether to use multiprocess logging.
    :rtype: list
    :return: a list with the results of executing the given function over each of the arguments. Indices will be
    aligned with the input arguments.
    """

    # check processes
    if processes == -1:
        processes = os.cpu_count()
    processes = min(processes, len(args), os.cpu_count())

    # check multiprocess logging
    if mp_logging:
        # redirects function to _log_processor to assign log handler
        assert MultiProcessLogger.queue is not None, 'MultiProcessLogger has not been created'
        args = [(func, _a, MultiProcessLogger.queue) for _a in args]
        func = _log_processor

    star = isinstance(args[0], tuple)  # star if function is multi-argument

    with parallel_backend('loky', inner_max_num_threads=os.cpu_count() // processes):  # spread cpus per job
        return _ProgressParallel(n_jobs=processes, use_tqdm=use_tqdm, total=len(args))(
            delayed(func)(*(arg if star else [arg])) for arg in args)


def _log_processor(func: Callable, args, queue: mp.Queue):
    # set root logger of this process to redirect to queue
    create_mp_log_handler(queue)

    # override singleton queue in this process to allow for nested calls to `run_parallel`
    MultiProcessLogger.queue = queue

    # execute function
    star = isinstance(args, tuple)  # star if function is multi-argument
    func(*(args if star else [args]))
