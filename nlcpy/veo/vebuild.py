#
# * The source code in this file is based on the soure code of PyVEO.
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
import collections
import subprocess
import os
from shutil import rmtree


GLOBAL_SO_FLAGS = "-shared -pthread"
MK_VEORUN_STATIC = "/opt/nec/ve/libexec/mk_veorun_static"

SUFFIX = {"C": ".c", "C++": ".cpp", "F": ".f90"}
COMPILER = {"C": "/opt/nec/ve/bin/ncc",
            "C++": "/opt/nec/ve/bin/nc++",
            "F": "/opt/nec/ve/bin/nfort"}
FLAGS = {
    "C": "-O2 -fpic -pthread -report-all -fdiag-vector=2",
    "C++": "-O2 -fpic -pthread -finline -finline-functions -report-all -fdiag-vector=2",
    "F": "-O2 -fpic -pthread -report-all -fdiag-vector=2"
}


def _shell_cmd(cmd, verbose=False):
    if verbose:
        print(cmd)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        if verbose:
            print(out)

    except Exception as e:
        print("%r" % e)
        print("Command failed: %s" % e.cmd)
        print("Output:\n%s" % e.output)
        return False
    return True


class VeObj(object):
    def __init__(self, content, flags=None, compiler=None):
        self._src = content
        # _type must be set here
        if flags is None:
            self._flags = FLAGS[self._type]
        else:
            self._flags = flags
        if compiler is None:
            self._compiler = COMPILER[self._type]
        else:
            self._compiler = compiler

    def build(self, oname, dir=None, verbose=False):
        owd = os.getcwd()
        if dir:
            os.chdir(dir)
        sname = oname + SUFFIX[self._type]
        with open(sname, "w") as f:
            f.write(self._src)

        cmd = self._compiler + " " + self._flags + " -c " + sname + " -o " + oname + ".o"
        try:
            rc = _shell_cmd(cmd, verbose=verbose)
        except Exception as e:
            os.chdir(owd)
            raise(e)
        if verbose and os.path.isfile(oname + ".L"):
            print("---------")
            with open(oname + ".L", "r") as f:
                for line in f:
                    if line.endswith("\n"):
                        line = line[:-1]
                    print(line)
            print("---------")
        if dir:
            os.chdir(owd)
        return rc

    def clean(self, oname):
        sname = oname + SUFFIX[self._type]
        try:
            os.unlink(sname)
            os.unlink(oname + ".o")
        except Exception:
            pass

    def get_compiler(self):
        return self._compiler

    def set_compiler(self, compiler):
        self._compiler = compiler

    def set_flags(self, flags):
        self._flags = flags

    def get_type(self):
        return self._type


class VeCObj(VeObj):
    def __init__(self, content, flags=None, compiler=None):
        self._type = "C"
        super(VeCObj, self).__init__(content, flags=flags, compiler=compiler)


class VeCppObj(VeObj):
    def __init__(self, content, flags=None, compiler=None):
        self._type = "C++"
        super(VeCppObj, self).__init__(content, flags=flags, compiler=compiler)


class VeFtnObj(VeObj):
    def __init__(self, content, flags=None, compiler=None):
        self._type = "F"
        super(VeFtnObj, self).__init__(content, flags=flags, compiler=compiler)


class VeBuild(object):
    _built = []  # built files, to be deleted by realclean()

    def __init__(self):
        self._obj = collections.OrderedDict()
        self._blddir = ""

    def set_c_src(self, label, content, flags=None, compiler=None):
        self._obj[label] = VeCObj(content, flags=flags, compiler=compiler)

    def set_cpp_src(self, label, content, flags=None, compiler=None):
        self._obj[label] = VeCppObj(content, flags=flags, compiler=compiler)

    def set_ftn_src(self, label, content, flags=None, compiler=None):
        self._obj[label] = VeFtnObj(content, flags=flags, compiler=compiler)

    def set_build_dir(self, dirname):
        if not dirname.endswith("/"):
            dirname = dirname + "/"
        self._blddir = dirname

    def build_so(self, label=None, flags=None, libs=[], linker=None, verbose=False):
        if label is None and self._obj.keys():
            label = self._first_label()
        if label is None:
            raise ValueError("No label. Did you define any sources?")
        self._check_create_blddir()
        for src, obj in self._obj.items():
            if obj.build(src, dir=self._blddir, verbose=verbose):
                print("compile %s -> ok" % src)
            else:
                print("compile %s -> failed" % src)
                return

        soname = self._blddir + label + ".so"
        if linker is None:
            linker = self._find_linker()
        if flags:
            cmd = linker + " " + flags
        else:
            cmd = linker + " " + GLOBAL_SO_FLAGS
        cmd += " -o " + soname
        cmd += " " + " ".join(
            ["%s%s.o" % (self._blddir, src) for src in self._obj.keys()]
        )
        if libs:
            cmd += " " + " ".join(libs)
        if _shell_cmd(cmd, verbose=verbose):
            if soname not in self._built:
                self._built.append(soname)
            return soname
        return None

    def build_veorun(self, label=None, flags=None, libs=[], verbose=False):
        if label is None and self._obj.keys():
            label = self._first_label()
        if label is None:
            raise ValueError("No label. Did you define any sources?")
        self._check_create_blddir()
        for src, obj in self._obj.items():
            if obj.build(src, dir=self._blddir, verbose=verbose):
                print("compile %s -> ok" % src)
            else:
                print("compile %s -> failed" % src)
                return

        oname = self._blddir + label + ".veorun"
        cmd = MK_VEORUN_STATIC + " " + oname
        if flags:
            cmd = "env CFLAGS=\"%s\" " % flags + cmd
        cmd += " " + " ".join(
            ["%s%s.o" % (self._blddir, src) for src in self._obj.keys()]
        )
        if libs:
            cmd += " " + " ".join(libs)
        if _shell_cmd(cmd, verbose=verbose):
            if oname not in self._built:
                self._built.append(oname)
            return oname
        return None

    def clean(self):
        for src, obj in self._obj.items():
            obj.clean(self._blddir + src)

    def clear(self):
        self.clean()
        self._obj = collections.OrderedDict()
        self._blddir = ""

    def realclean(self):
        self.clean()
        for file in self._built:
            try:
                os.unlink(file)
                self._built.remove(file)
            except Exception:
                pass
        if self._blddir != "":
            rmtree(self._blddir)

    def _check_create_blddir(self):
        if self._blddir == "":
            return
        if os.path.isdir(self._blddir):
            if not os.access(self._blddir, os.W_OK | os.R_OK):
                raise OSError("Directory '%s' exists but is not readable "
                              "or writable!" % self._blddir)
        else:
            os.mkdir(self._blddir)

    def _first_label(self):
        if self._obj.keys():
            for k in self._obj.keys():
                return k
        return None

    def _find_linker(self):
        _type = None
        for src, obj in self._obj.items():
            otype = obj.get_type()
            if _type is None:
                _type = otype
                continue
            if otype == "C++":
                _type = "C++"
            elif otype == "F":
                if _type == "C":
                    _type = "F"
        if _type is not None:
            return COMPILER[_type]
        raise ValueError("No linker. Did you define any sources?")
