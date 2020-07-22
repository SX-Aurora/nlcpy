
# NLCPy : NumPy script accelerator for SX-Aurora TSUBASA

*NLCPy* is a package for accelerating performance of Python scripts using `NumPy` on SX-Aurora TSUBASA. Python programmers can use this package on Linux/x86 of SX-Aurora TSUBASA. NLCPy API is designed based on NumPy one. The current version provides a subset of NumPy API.

### Requirements

Before the installation, the following components are required to be installed on your x86 Node of SX-Aurora TSUBASA.

- [NEC SDK](https://www.hpc.nec/documents/guide/pdfs/InstallationGuide_E.pdf)
	- required NEC C/C++ compiler version: >= 3.0.4
	- required NLC version: >= 2.1.0

- [Alternative VE Offloading (AVEO)](https://veos-sxarr-nec.github.io/aveo/index.html)
	- If you install NLCpy from wheel, the runtime packages of Alternative VE Offloading(AVEO) are required. 
    
        ```
        # yum install veoffload-aveo veoffload-aveorun
        ```

	- If you install NLCpy from source, the development packages of Alternative VE Offloading(AVEO) are required. 
	
        ```
        # yum install veoffload-aveo-devel veoffload-aveorun-devel
        ```  

- [Python](https://www.python.org/)
	- required version: >=3.6

### Install from wheel 

You can install NLCPy by executing either of following commands. 

- Install from PyPI

    ```
    $ pip install nlcpy
    ```

- Install from your local computer

    1. Download the wheel package from [GitHub](https://github.com/SX-Aurora/nlcpy/).

    2. Put the wheel package to your any directory. 

    3. Install the local wheel package via pip command.  
    
        ```
        $ pip install <path_to_wheel>
        ```

The shared objects for Vector Engine, which are included in the wheel package, are compiled and tested by using NEC C/C++ Version 3.0.4.

### Install from source (with building)

Before building source files, please install following packages.   

```
$ pip install numpy cython wheel
```  

And, entering these commands in the environment where gcc and ncc commands are available.  

```
$ git clone https://github.com/SX-Aurora/nlcpy.git
$ cd nlcpy
$ pip install .
```  

## Documentation
- [NLCPy User's Guide](https://www.hpc.nec/documents/nlcpy/en/index.html)

## License

NLCPy is available as free software under the 3-clause BSD license. (see `LICENSE` file).  

