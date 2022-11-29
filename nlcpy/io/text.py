#
# * The source code in this file is developed independently by NEC Corporation.
#
# # NLCPy License #
#
#     Copyright (c) 2020 NEC Corporation
#     All rights reserved.
#
#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither NEC Corporation nor the names of its contributors may be
#       used to endorse or promote products derived from this software
#       without specific prior written permission.
#
#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#     DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#     FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from nlcpy.wrapper.numpy_wrap import numpy_wrap


@numpy_wrap
def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='n', header='', footer='',
            comments='# ', encoding=None):
    """Saves an array to a text file.

    Parameters
    ----------
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in compressed
        gzip format. :func:`loadtxt` understands gzipped files transparently.
    x : 1D or 2D array_like
        Data to be saved to a text file.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a multi-format string, e.g.
        'Iteration %d – %10.5f', in which case *delimiter* is ignored.
        For complex *X*, the legal options for *fmt* are:

        - a single specifier, *fmt='%.4e'*, resulting in numbers formatted like
          *'(%s+%sj)' % (fmt, fmt)*
        - a full string specifying every real and imaginary part,
          e.g. *'%.4e %+.4ej %.4e %+.4ej %.4e %+.4ej'* for 3 columns
        - a list of specifiers, one per column - in this case, the real and imaginary
          part must have separate specifiers, e.g. *['%.3e + %.3ej', '(%.15e%+.15ej)']*
          for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
    header : str, optional
        String that will be written at the beginning of the file.
    footer : str, optional
        String that will be written at the end of the file.
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ', as expected by e.g. ``numpy.loadtxt``.
    encoding : {None, str}, optional
        Encoding used to encode the outputfile. Does not apply to output streams.
        If the encoding is something other than 'bytes' or 'latin1' you will not be
        able to load the file in NumPy versions < 1.14. Default is 'latin1'.

    See Also
    --------
    save : Saves an array to a binary file in NumPy ``.npy`` format.
    savez : Saves several arrays into a single file in uncompressed ``.npz`` format.
    savez_compressed : Saves several arrays into a single file in compressed
                       ``.npz`` format.

    Note
    ----
    Further explanation of the fmt parameter (``%[flag]width[.precision]specifier``):

    **flags:**
        ``-`` : left justify

        ``+`` : Forces to precede result with + or -.

        ``0`` : Left pad the number with zeros instead of space (see width).

    **width:**
        Minimum number of characters to be printed.
        The value is not truncated if it has more characters.

    **precision:**
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print after
          the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.

    **specifiers:**
        ``c`` : character

        ``d`` or ``i`` : signed decimal integer

        ``e`` or ``E`` : scientific notation with ``e`` or ``E``

        ``f`` : decimal floating point

        ``g,G`` : use the shorter of e,E or f

        ``o`` : signed octal

        ``s`` : string of characters

        ``u`` : unsigned decimal integer

        ``x,X`` : unsigned hexadecimal integer

    This explanation of ``fmt`` is not complete, for an exhaustive specification
    see `Format Specification Mini-Language,
    <https://docs.python.org/3/library/string.html#format-specification-mini-language>`_
    Python Documentation.

    Examples
    --------
    >>> import nlcpy as vp
    >>> x = y = z = vp.arange(0.0, 5.0, 1.0)
    >>> vp.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> vp.savetxt('test.out', (x, y, z))   # x,y,z equal sized 1D arrays
    >>> vp.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
    """
    # use numpy_wrap
    raise NotImplementedError('savetxt is not implemented yet.')
