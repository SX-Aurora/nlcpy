import datetime
import os


def pytest_addoption(parser):
    parser.addoption(
        "--test", default='no_ufunc')


def pytest_configure(config):
    if not config.option.resultlog:
        timestamp = datetime.datetime.strftime(
            datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')
        filepath = os.path.dirname(os.path.abspath(__file__)) + \
            '/result/pytest_all_log.' + timestamp
        config.option.resultlog = filepath
