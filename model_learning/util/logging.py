import logging
import logging.handlers
import multiprocessing as mp

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'


def change_log_handler(log_file: str,
                       level: int = logging.WARN,
                       append: bool = False,
                       fmt: str = '[%(asctime)s %(levelname)s] %(message)s'):
    """
    Changes root logger to log to given file and to the console.
    :param str log_file: the path to the intended log file.
    :param bool append: whether to append to the log file, if it exists already.
    :param int level: the level of the log messages below which will be saved to file.
    :param str fmt: the formatting string for the messages.
    :return:
    """
    root = logging.getLogger()
    # for handler in log.handlers[:]:
    #     log.removeHandler(handler)
    file_handler = logging.FileHandler(log_file, 'a' if append else 'w')
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.level = file_handler.level = stream_handler.level = level


class MultiProcessLogger(object):
    """
    A multiprocess logging handler allowing logging to a file and to the console from multiple processes.
    Interested worker processes should create a `QueueHandler` using the `MultiProcessLogger.queue` singleton instance,
    e.g., by using the `create_mp_log_handler` function.
    See: https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
    See: https://fanchenbao.medium.com/python3-logging-with-multiprocessing-f51f460b8778
    """
    queue: mp.Queue = None  # singleton

    def __init__(self,
                 log_file: str,
                 level: int = logging.WARN,
                 append: bool = False,
                 fmt: str = '[%(asctime)s %(levelname)s] %(message)s'):
        """
        Creates a new multiprocess logger.
        :param str log_file: the path to the intended log file.
        :param bool append: whether to append to the log file, if it exists already.
        :param int level: the level of the log messages below which will be saved to file.
        :param str fmt: the formatting string for the messages.
        """

        assert self.queue is None, f'Another MultiProcessLogger has been created!'

        # creates queue and log listener process
        m = mp.Manager()
        MultiProcessLogger.queue = m.Queue()
        self.listener = mp.Process(target=self._listener, args=(log_file, level, append, fmt))
        self.listener.start()

    def _listener(self, log_file: str, level: int, append: bool, fmt: str):
        change_log_handler(log_file, level, append, fmt)
        while True:
            record = MultiProcessLogger.queue.get()
            if record is None:  # waits for None to exit log listener process
                return
            logger = logging.getLogger(record.name)
            logger.handle(record)

    def close(self):
        MultiProcessLogger.queue.put(None)  # tells listener to quit
        self.listener.join()


def create_mp_log_handler(queue: mp.Queue):
    """
    Modifies the root logger to redirect log messages via a `QueueHandler` with the given queue.
    Useful to redirect log messages in a child process.
    :param mp.Queue queue: the queue to which log messages are to be sent for handling.
    """
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)  # everything is sent to the handler
