.. module:: nlcpy.prof

Profiling Routines
==================

The following table shows profiling routines provided by NLCPy.

.. _label_profiling_ftrace:

Profiling with FTRACE
---------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.prof.ftrace_region
    nlcpy.prof.ftrace_region_begin
    nlcpy.prof.ftrace_region_end

FTRACE is a performance analysis tool, which can obtain performance information
such as the exclusive time and vectorization aspect on the VE.
Please note that VE offloading overhead is included in performance information.
For details of FTRACE, see
`PROGINF/FTRACE User's Guide <https://www.hpc.nec/documents/sdk/pdfs/g2at03e-PROGINF_FTRACE_User_Guide_en.pdf>`_.

.. note::

    VE functions in NLCPy are multithreaded.
    However, the ftrace.out includes performance information of only a master thread.
    To obtain performance information of all threads, it is necessary to compile
    C programs of NLCPy by using ncc with the option '-ftrace'.
    If needed, please download source programs of NLCPy from GitHub.


|
|
| **The following routines are deprecated since NLCPy version 2.0.0.**

Start and Stop Profiling
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.prof.start_profiling
    nlcpy.prof.stop_profiling

Get Result
----------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.prof.get_run_stats
    nlcpy.prof.print_run_stats
