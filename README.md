![NLCPy_banner](https://github.com/SX-Aurora/nlcpy/blob/master/banner/NLCPy_banner.png?raw=true)

# NLCPy : NumPy-like API accelerated with SX-Aurora TSUBASA

[![GitHub license](https://img.shields.io/github/license/SX-Aurora/nlcpy)](https://github.com/SX-Aurora/nlcpy/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/nlcpy)](https://pepy.tech/project/nlcpy)
[![Python Versions](https://img.shields.io/pypi/pyversions/nlcpy.svg)](https://pypi.org/project/nlcpy/)


`NLCPy` is a library for accelerating performance of Python scripts using `NumPy` on SX-Aurora TSUBASA. Python programmers can use this library on Linux/x86 of SX-Aurora TSUBASA. NLCPy's API is designed based on NumPy's one. The current version provides a subset of NumPy's API.

## Requirements

Before the installation, the following components are required to be installed on your x86 Node of SX-Aurora TSUBASA.

- [NEC SDK](https://sxauroratsubasa.sakura.ne.jp/documents/guide/pdfs/InstallationGuide_E.pdf)
	- required NEC C/C++ compiler version: >= 5.0.0
	- required NLC version: >= 3.0.0

- [Alternative VE Offloading (AVEO)](https://sxauroratsubasa.sakura.ne.jp/documents/veos/en/aveo/index.html)

    - required version: >= 3.0.2
	- If you install NLCPy from wheel, the runtime packages of Alternative VE Offloading(AVEO) are required.

        ```
        # yum install veoffload-aveo veoffload-aveorun-ve1 veoffload-aveorun-ve3
        ```

	- If you install NLCPy from source, the development packages of Alternative VE Offloading(AVEO) are required.

        ```
        # yum install veoffload-aveo-devel veoffload-aveorun-ve1-devel veoffload-aveorun-ve3-devel
        ```

- veosinfo3

    ```
    # yum install veosinfo3
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

    1. Download [the wheel package](https://github.com/SX-Aurora/nlcpy/releases) from GitHub.

    2. Put the wheel package to your any directory.

    3. Install the local wheel package via pip command.

        ```
        $ pip install <path_to_wheel>
        ```

The shared objects for Vector Engine, which are included in the wheel package, are tested by using NEC C/C++ Compiler Version 5.0.0 and NumPy v1.19.5.

## Install from source (with building)

Before building source files, please install following packages.

```
$ pip install numpy cython wheel
$ sudo yum install veosinfo3-devel
$ sudo yum install veoffload-aveo-devel veoffload-aveorun-ve1-devel veoffload-aveorun-ve3-devel
```

And, entering these commands in the environment where `gcc` and `ncc` commands are available.

```
$ git clone https://github.com/SX-Aurora/nlcpy.git
$ cd nlcpy
$ python3 setup.py build_ext --targ ve1,ve3,vh
$ python3 setup.py intall --targ ve1,ve3,vh
```

## Documentation
- [NLCPy User's Guide](https://sxauroratsubasa.sakura.ne.jp/documents/nlcpy/en/index.html)

## License

The BSD-3-Clause license (see `LICENSE` file).

NLCPy is derived from NumPy, CuPy, PyVEO, and numpydoc (see `LICENSE_DETAIL/LICENSE_DETAIL` file).
