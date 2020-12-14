def pytest_addoption(parser):
    parser.addoption(
        "--test", default='no_ufunc')
