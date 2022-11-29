import logging
import sys
import atexit

VEO = 'nlcpy.veo'
MEMPOOL = 'nlcpy.mempool'
REQUEST = 'nlcpy.request'
NPWRAP = 'nlcpy.numpy_wrap'
FFT = 'nlcpy.fft'

_exitmode = False


def _get_logger(name):
    _logger = logging.getLogger(name)
    _logger.propagate = False
    _logger.addHandler(logging.NullHandler())
    return _logger


_loggers = {
    VEO: [_get_logger(VEO), False],
    MEMPOOL: [_get_logger(MEMPOOL), False],
    REQUEST: [_get_logger(REQUEST), False],
    NPWRAP: [_get_logger(NPWRAP), False],
    FFT: [_get_logger(FFT), False],
}


def get_logger(logger_name):
    """ Gets the logger.

    Parameters
    ----------
    logger_name : str
        Logger name

    Returns
    -------
    logger : logger
        Logger that corresponds to the logger name.

    """

    global _loggers
    return _loggers[logger_name][0]


def _is_enable(logger_name):
    global _loggers
    return _loggers[logger_name][1]


def _set_flag(logger_name, flag):
    global _loggers
    _loggers[logger_name][1] = flag


def set_stream_handler(logger_name, stream=sys.stderr, level=logging.INFO, format=None):
    """ Sets StreamHandler to the specified logger.

    Parameters
    ----------
    logger_name : str
        Logger name
    stream : stream
        Stream to be logged.
    level : level
        Logging level.
    format : format
        Logging format.

    Returns
    -------
    logger : logger
        Logger that corresponds to the logger name.

    """

    _logger = get_logger(logger_name)

    if format is None:
        format = logging.BASIC_FORMAT

    _logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    _set_flag(logger_name, True)

    return _logger


def set_file_handler(logger_name, filename, mode='a', encoding=None, delay=False,
                     level=logging.INFO, format=None):
    """ Sets FileHandler to the specified logger.

    Parameters
    ----------
    logger_name : str
        Logger name
    filename : str
        File name to be logged.
    mode : mode
        File access mode.
    encoding : encoding
        Encoding.
    delay : bool
        Delay mode.
    level : level
        Logging level.
    format : format
        Logging format.

    Returns
    -------
    logger : logger
        Logger that corresponds to the logger name.

    """

    _logger = get_logger(logger_name)

    if format is None:
        format = logging.BASIC_FORMAT

    _logger.setLevel(level)
    handler = logging.FileHandler(filename, mode=mode, encoding=encoding, delay=delay)
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

    _set_flag(logger_name, True)

    return _logger


def reset_handler(logger_name, level=logging.WARN):
    """ Resets the logger.

    Parameters
    ----------
    logger_name : str
        Logger name
    level : level
        Logging level.

    Returns
    -------
    logger : logger
        Logger that corresponds to the logger name.

    """

    _logger = get_logger(logger_name)

    _logger.handlers.clear()
    _logger.addHandler(logging.NullHandler())

    _set_flag(logger_name, False)

    return _logger


# TODO: show caller code and lineno of user code.

def debug(logger_name, msg, *args, **kwargs):
    if not _exitmode:
        get_logger(logger_name).debug(msg, *args, **kwargs)


def info(logger_name, msg, *args, **kwargs):
    if not _exitmode:
        get_logger(logger_name).info(msg, *args, **kwargs)


@atexit.register
def _logging_finalize():
    # Do not log into FileHandler during Python finalization.
    # See https://bugs.python.org/issue26789.
    global _exitmode
    _exitmode = True
