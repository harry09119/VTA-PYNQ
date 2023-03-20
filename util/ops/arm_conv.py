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
Compile PyTorch Models
======================
**Author**: `Alex Wong <https://github.com/alexwong/>`_

This article is an introductory tutorial to deploy PyTorch models with Relay.

For us to begin, PyTorch should be installed.
TorchVision is also required so we can use the model zoo.
A quick solution is to install via pip:

.. code-block:: bash

    %%shell
    pip install torch
    pip install torchvision

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.7 and 1.4. Other versions may
be unstable.
"""
import sys
import os
from time import time

import tvm
from tvm import relay
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download

import numpy as np

from tvm.contrib.download import download_testdata
import vta
from vta.testing import simulator
from vta.top import graph_pack

# PyTorch imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Load  a PyTorch model
# -------------------------------
ic, oc, is_, ks, st = [16, 32, 32, 3, 1]

class conv(nn.Module):
    def __init__(self,ic,oc,ks,st):
        super(conv, self).__init__()
        self.layer = nn.Conv2d(in_channels=ic,out_channels=oc,kernel_size=ks,stride=st,bias=False)

    def forward(self,x):
        x = self.layer(x)
        return x

env = vta.get_env()
device = "arm_cpu"

target = env.target if device == "vta" else env.target_vta_cpu


tracker_host = os.environ.get("TVM_TRACKER_HOST", "10.201.135.166")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 8104))

model = conv(ic,oc,ks,st)
input_shape = [1, ic, is_, is_]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

######################################################################
# Import the graph to Relay
# -------------------------
# Convert PyTorch graph to Relay graph. The input name can be arbitrary.
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
if env.TARGET != "sim":
    # Get remote from fleet node
    remote = autotvm.measure.request_remote(
        env.TARGET, tracker_host, tracker_port, timeout=10000
    )
    # Reconfigure the JIT runtime and FPGA.
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)
else:
    # In simulation mode, host the RPC server locally.
    remote = rpc.LocalSession()

with tvm.transform.PassContext(opt_level=2, disabled_pass={"AlterOpLayout"}):
    lib = relay.build(
        mod, target=target, params=params, target_host=env.target_host
    )

# Export library
print("Upload...")
temp = utils.tempdir()
lib.export_library(temp.relpath("graphlib.tar"))
remote.upload(temp.relpath("graphlib.tar"))
lib = remote.load_module("graphlib.tar")

# Generate the graph executor
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
#m = graph_executor.GraphModule(lib["default"](ctx))
m = graph_executor.create(graph, lib,ctx)

# upload parameters to device
image = tvm.nd.array((np.random.uniform(size=(1, ic, is_, is_))).astype("float32"))
m.set_input("input0", image)

# evaluate
print("Evaluate inference time cost...")
timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
tcost = timer()
prof_res = np.array(tcost.results) * 1000  # convert to millisecond
print(
    "Mean inference time (std dev): %.2f ms (%.2f ms)"
    % (np.mean(prof_res), np.std(prof_res))
)

