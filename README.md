![NLCPy_banner](https://github.com/SX-Aurora/nlcpy/blob/master/banner/NLCPy_banner.png?raw=true)

# NLCPy : NumPy-like API accelerated with SX-Aurora TSUBASA

[![GitHub license](https://img.shields.io/github/license/SX-Aurora/nlcpy)](https://github.com/SX-Aurora/nlcpy/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/nlcpy)](https://pepy.tech/project/nlcpy)
[![Python Versions](https://img.shields.io/pypi/pyversions/nlcpy.svg)](https://pypi.org/project/nlcpy/)


`NLCPy` is a library for accelerating performance of Python scripts using `NumPy` on SX-Aurora TSUBASA. Python programmers can use this library on Linux/x86 of SX-Aurora TSUBASA. NLCPy's API is designed based on NumPy's one. The current version provides a subset of NumPy's API.

## Requirements

Before the installation, the following components are required to be installed on your x86 Node of SX-Aurora TSUBASA.

- [NEC SDK](https://www.hpc.nec/documents/guide/pdfs/InstallationGuide_E.pdf)
	- required NEC C/C++ compiler version: >= 3.3.1
	- required NLC version: >= 2.3.0

- [Alternative VE Offloading (AVEO)](https://www.hpc.nec/documents/veos/en/aveo/index.html)

	- If you install NLCpy from wheel, the runtime packages of Alternative VE Offloading(AVEO) are required.

        ```
        # yum install veoffload-aveo veoffload-aveorun
        ```

	- If you install NLCpy from source, the development packages of Alternative VE Offloading(AVEO) are required.

        ```
        # yum install veoffload-aveo-devel veoffload-aveorun-devel
        ```

- [Python](https://www.python.org/)
        - required version: 3.6, 3.7, or 3.8

- [NumPy](https://www.numpy.org/)
        - required version: >= v1.17

## Install from wheel

You can install NLCPy by executing either of following commands.

- Install from PyPI

    ```
    $ pip install nlcpy
    ```

- Install from your local computer

    1. Download [the wheel package](https://github.com/SX-Aurora/nlcpy/releases/tag/v1.0.0) from GitHub.

    2. Put the wheel package to your any directory.

    3. Install the local wheel package via pip command.

        ```
        $ pip install <path_to_wheel>
        ```

The shared objects for Vector Engine, which are included in the wheel package, are compiled and tested by using NEC C/C++ Version 3.3.1 and NumPy v1.19.2.

## Install from source (with building)

Before building source files, please install following packages.

```
$ pip install numpy cython wheel
```

And, entering these commands in the environment where `gcc` and `ncc` commands are available.

```
$ git clone https://github.com/SX-Aurora/nlcpy.git
$ cd nlcpy
$ pip install .
```

## Documentation
- [NLCPy User's Guide](https://www.hpc.nec/documents/nlcpy/en/index.html)

## License

The BSD-3-Clause license (see `LICENSE` file).

NLCPy is derived from NumPy, CuPy, PyVEO, and numpydoc (see `LICENSE_DETAIL/LICENSE_DETAIL` file).
