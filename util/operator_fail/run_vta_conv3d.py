# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _vta-get-started:

Get Started with VTA
====================
**Author**: `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

This is an introduction tutorial on how to use TVM to program the VTA design.

In this tutorial, we will demonstrate the basic TVM workflow to implement
a vector addition on the VTA design's vector ALU.
This process includes specific scheduling transformations necessary to lower
computation down to low-level accelerator operations.

To begin, we need to import TVM which is our deep learning optimizing compiler.
We also need to import the VTA python package which contains VTA specific
extensions for TVM to target the VTA design.
"""
from __future__ import absolute_import, print_function

import os
import tvm
from tvm import te
import vta
import numpy as np

env = vta.get_env()

from tvm import rpc
from tvm.contrib import utils
from vta.testing import simulator
from time import time

host = os.environ.get("VTA_RPC_HOST", "10.201.135.166")
port = int(os.environ.get("VTA_RPC_PORT", "8105"))

if env.TARGET == "pynq" or env.TARGET == "de10nano":

    # Make sure that TVM was compiled with RPC=1
    assert tvm.runtime.enabled("rpc")
    remote = rpc.connect(host, port)

    # Reconfigure the JIT runtime
    vta.reconfig_runtime(remote)

    # Program the FPGA with a pre-compiled VTA bitstream.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    vta.program_fpga(remote, bitstream=None)

# In simulation mode, host the RPC server locally.
elif env.TARGET in ("sim", "tsim", "intelfocl"):
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

input_s = (1,32,32,32,1,16)
kernel_s = (1,2,3,3,16,16)
output_s = (1,32,30,30,1,16)
data = te.placeholder(input_s, name="data", dtype=env.inp_dtype)
kernel = te.placeholder(kernel_s, name="kernel", dtype=env.wgt_dtype)

# A copy buffer
data_buf = te.compute(input_s, lambda *i: data(*i), "data_buf")
# B copy buffer
kernel_buf = te.compute(kernel_s, lambda *i: kernel(*i), "kernel_buf")

# Describe the in-VTA vector addition

di = te.reduce_axis((0, kernel_s[2]), name="di")#=14
dj = te.reduce_axis((0, kernel_s[3]), name="dj")#=14
ko = te.reduce_axis((0, input_s[1]), name="ko") #=16
ki = te.reduce_axis((0, input_s[-1]), name="ki")#=16

#input_s = (1,16,14,14,16)    = (b_o, k_o, i, j, b_i, k_i)
#kernel_s = (32,16,1,1,16,16) = (c_o, k_o, d_i, d_j, c_i, k_i)
#output_s = (1,32,14,14,1,16) = (b_o, c_o, i, j, b_i, c_i)

store_buf = te.compute(
                        output_s,
                        lambda bo, co, i, j, bi, ci: te.sum(
                            data_buf[bo, ko, i + di, j + dj, bi, ki].astype(env.acc_dtype)
                            * kernel_buf[co, ko, di, dj, ci, ki].astype(env.acc_dtype),
                            axis=[ko, di, dj, ki],
                        ),
                        name="store_buf",
                      )

# Cast to output type, and send to main memory
output = te.compute(
                    output_s,
                    lambda *i: store_buf(*i).astype(env.out_dtype), name="output"
)

# Let's take a look at the generated schedule
s = te.create_schedule(output.op)

print(tvm.lower(s, [data, kernel, output], simple_mode=True))

#-----SCHEDULE-----

#, c_o, x_i, x_j, _, _ = s[store_buf].op.axis
#_i, _, _, _ = s[store_buf].op.reduce_axis

# Set the intermediate tensors' scope to VTA's on-chip accumulator buffer
s[data_buf].set_scope(env.inp_scope)
s[kernel_buf].set_scope(env.wgt_scope)
s[store_buf].set_scope(env.acc_scope)
s[output].set_scope(env.acc_scope)

#reorder
x_bo, x_co, x_i, x_j, x_bi, x_ci = s[output].op.axis
x_co0, x_co1 =  s[output].split(x_co,factor=4)
x_i0, x_i1 = s[output].split(x_i,factor=32)
x_j0, x_j1 = s[output].split(x_j,factor=32)
s[output].reorder(x_bo, x_i0, x_co0, x_j0, x_co1, x_i1, x_j1, x_bi, x_ci)

x_bo, x_co, x_i, x_j, x_bi, x_ci = s[store_buf].op.axis
k_o, d_i, d_j, k_i = s[store_buf].op.reduce_axis
#print(k_o, d_i, d_j, k_i)
#x_co0, x_co1 =  s[store_buf].split(x_co,factor=4)
#k_o0, k_o1 = s[store_buf].split(k_o,factor=4)

s[store_buf].reorder(x_bo, k_o, x_j, d_j, d_i, x_co, x_i, x_bi, x_ci, k_i)

ko, _ = s[store_buf].split(ko, factor=4)

#Move buffer copy into conv2d loop
s[data_buf].compute_at(s[store_buf], ko)
s[kernel_buf].compute_at(s[store_buf], ko)
s[store_buf].compute_at(s[output], x_j0)
#s[store_buf].compute_at(s[output], s[output].op.axis[3])
print(tvm.lower(s, [data, kernel, output], simple_mode=False))

# DMA transfer operation
s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
s[kernel_buf].pragma(s[kernel_buf].op.axis[0], env.dma_copy)
print(tvm.lower(s, [data, kernel, output], simple_mode=False))

s[store_buf].tensorize(s[store_buf].op.axis[4],env.gemm)
s[output].pragma(s[output].op.axis[4], env.dma_copy)

print(tvm.lower(s, [data, kernel, output], simple_mode=False))

#print(tvm.lower(s, [A, B, C], simple_mode=True))

# Let's take a look at the finalized schedule
print(vta.lower(s, [data, kernel, output], simple_mode=True))

conv2d = vta.build(
    s, [data, kernel, output], tvm.target.Target("ext_dev", host=env.target_host), name="conv2d"
)

# Write the compiled module into an object file.
temp = utils.tempdir()
conv2d.save(temp.relpath("conv2d.o"))

# Send the executable over RPC
remote.upload(temp.relpath("conv2d.o"))

# Loading the Module
f = remote.load_module("conv2d.o")

# Get the remote device context
ctx = remote.ext_dev(0)

# Initialize the A and B arrays randomly in the int range of (-128, 128]
data_orig = np.random.randint(-128, 128, size=(1,32*16,14,14)).astype(data.dtype)
kernel_orig = np.random.randint(-128, 128, size=(32*16,32*16,1,1)).astype(kernel.dtype)

# Apply packing to the A and B arrays from a 2D to a 4D packed layout
data_packed = data_orig.reshape(1, 32, 1, 16, 14, 14).transpose((0, 1, 4, 5, 2, 3))
kernel_packed = kernel_orig.reshape(32,32,16,16,1,1).transpose((0, 1, 4, 5, 2, 3))

# Format the input/output arrays with tvm.nd.array to the DLPack standard
data_nd = tvm.nd.array(data_packed, ctx)
kernel_nd = tvm.nd.array(kernel_packed, ctx)
output_nd = tvm.nd.array(np.zeros((1, 32, 32, 32, 1, 16)).astype(output.dtype), ctx)

# Invoke the module to perform the computation
start=time()
f(data_nd, kernel_nd, output_nd)
print(time()-start)

# Compute reference result with numpy
print("Successful vector add test!")
