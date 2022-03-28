#
# * The source code in this file is based on the soure code of numpydoc.
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
# ## numpydoc License ##
#
#  Copyright (C) 2008 Stefan van der Walt <stefan@mentat.za.net>,
#  Pauli Virtanen <pav@iki.fi>
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#   1. Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
#  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
#  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES

"""
Implementing `python -m nlcpydoc` functionality.
"""

import sys
import argparse
import ast

from .docscrape_sphinx import get_doc_object
from .validate import validate, Docstring


def render_object(import_path, config=None):
    """Test numpydoc docstring generation for a given object"""
    # TODO: Move Docstring._load_obj to a better place than validate
    print(get_doc_object(Docstring(import_path).obj,
                         config=dict(config or [])))
    return 0


def validate_object(import_path):
    exit_status = 0
    results = validate(import_path)
    for err_code, err_desc in results["errors"]:
        exit_status += 1
        print(':'.join([import_path, err_code, err_desc]))
    return exit_status


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('import_path', help='e.g. numpy.ndarray')

    def _parse_config(s):
        key, _, value = s.partition('=')
        value = ast.literal_eval(value)
        return key, value

    ap.add_argument('-c', '--config', type=_parse_config,
                    action='append',
                    help='key=val where val will be parsed by literal_eval, '
                         'e.g. -c use_plots=True. Multiple -c can be used.')
    ap.add_argument('--validate', action='store_true',
                    help='validate the object and report errors')
    args = ap.parse_args()

    if args.validate:
        exit_code = validate_object(args.import_path)
    else:
        exit_code = render_object(args.import_path, args.config)

    sys.exit(exit_code)
