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


cdef extern from "<veosinfo/veosinfo.h>":
    enum: VE_EINVAL_COREID      # -514  Error number for invalid cores
    enum: VE_EINVAL_NUMAID      # -515  Error number for invalid NUMA node
    enum: VE_EINVAL_LIMITOPT    # -516  Error number for invalid VE_LIMIT_OPT
    enum: VE_ERANGE_LIMITOPT    # -517  Error number if limit is out of range
    enum: VEO_PROCESS_EXIST     #  515  Identifier for VEO API PID
    enum: VE_VALID_THREAD       #  516  Identifier for process/thread
    enum: VE_MAX_NODE           #  8    Maximum number of VE nodes
    enum: VE_PATH_MAX           # 4096
    enum: VE_FILE_NAME          # 255   Maximum length of file name
    enum: FILENAME              # 15    Length of file used to get
                                #       pmap/ipcs/ipcrm command information
    enum: VE_BUF_LEN            # 255
    enum: VE_MAX_CORE_PER_NODE  # 16
    enum: VMFLAGS_LENGTH        # 81
    enum: VE_MAX_CACHE          # 4  max number of VE caches
    enum: VE_DATA_LEN           # 20
    enum: VE_MAX_REGVALS        # 64
    enum: MAX_DEVICE_LEN        # 255
    enum: MAX_POWER_DEV         # 20
    enum: VE_PAGE_SIZE          # 2097152
    enum: VKB                   # 1024
    enum: EXECVE_MAX_ARGS       # 256
    enum: MICROSEC_TO_SEC       # 1000000
    enum: VE_NUMA_NUM           # 2  maximum number of NUMA nodes in a VE
    enum: MAX_CORE_IN_HEX       # 4  maximum core value in HEX
    enum: MAX_SWAP_PROCESS      # 256 maximum number of swapped processes
    DEF VE_EXEC_PATH = "/opt/nec/ve/bin/ve_exec"
    DEF VE_NODE_SPECIFIER = "-N"

    cdef struct ve_nodeinfo:
        int nodeid[VE_MAX_NODE]
        int status[VE_MAX_NODE]
        int cores[VE_MAX_NODE]
        int total_node_count

    struct ve_meminfo:
        unsigned long kb_main_total
        unsigned long kb_main_used
        unsigned long kb_main_free
        unsigned long kb_main_shared
        unsigned long kb_main_buffers
        unsigned long kb_main_cached
        unsigned long kb_swap_cached
        unsigned long kb_low_total
        unsigned long kb_low_free
        unsigned long kb_high_total
        unsigned long kb_high_free
        unsigned long kb_swap_total
        unsigned long kb_swap_free
        unsigned long kb_active
        unsigned long kb_inactive
        unsigned long kb_dirty
        unsigned long kb_committed_as
        unsigned long hugepage_total
        unsigned long hugepage_free
        unsigned long kb_hugepage_used

    int ve_check_node_status(int)
    int ve_node_info(ve_nodeinfo *)
    int ve_mem_info(int, ve_meminfo *)
