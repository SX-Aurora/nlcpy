Logging (for developer)
=======================

The following tables show logging routines provided by NLCPy.

Logging management
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    nlcpy.logging.set_stream_handler
    nlcpy.logging.set_file_handler
    nlcpy.logging.reset_handler


The following logger names are available in NLCPy.

Logger Constants
----------------

.. data:: nlcpy.logging.VEO
.. data:: nlcpy.logging.MEMPOOL
.. data:: nlcpy.logging.REQUEST
.. data:: nlcpy.logging.NPWRAP
.. data:: nlcpy.logging.FFT


Example Usage
-------------

::

    >>> import nlcpy
    >>> _ = nlcpy.logging.set_stream_handler(nlcpy.logging.VEO)
    >>> _ = nlcpy.logging.set_stream_handler(nlcpy.logging.REQUEST)
    >>> _ = nlcpy.logging.set_stream_handler(nlcpy.logging.MEMPOOL)
    >>> nlcpy.arange(10)
    INFO:nlcpy.mempool:nlcpy_mempool_reserve used: nodeid=0, addr=8000610054000010, size=80
    INFO:nlcpy.request:push VE request `nlcpy_arange` (nodeid=0)
    INFO:nlcpy.request:veo_async_write_mem to send VE arguments (nodeid=0)
    INFO:nlcpy.veo:veo_async_write_mem: nodeid=0, size=3568, reqid=155
    INFO:nlcpy.request:veo_call_async to flush stacked requests (nodeid=0): requests <nlcpy_arange>
    INFO:nlcpy.veo:veo_call_async: name=b'kernel_launcher', reqid=156
    INFO:nlcpy.veo:veo_call_wait_result: nodeid=0, reqid=155
    INFO:nlcpy.veo:veo_call_wait_result: nodeid=0, reqid=156
    INFO:nlcpy.veo:veo_read_mem: nodeid=0, size=80
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _ = nlcpy.logging.reset_handler(nlcpy.logging.VEO)
    >>> _ = nlcpy.logging.reset_handler(nlcpy.logging.REQUEST)
    >>> _ = nlcpy.logging.reset_handler(nlcpy.logging.MEMPOOL)
    >>> nlcpy.arange(10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


