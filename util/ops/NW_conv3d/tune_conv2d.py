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
Auto-tuning a convolutional network on VTA
==========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

Auto-tuning for a specific accelerator design is critical for getting the best
performance for any given operator. This is a tutorial showcases how to tune a
whole convolutional network on VTA.

The operator implementation for VTA in TVM is written in template form.
The template has many tunable knobs (tile factor, virtual threads, etc).
We will tune all convolution operators in the neural network. After tuning,
we produce a log file which stores the best schedule parameters for all tuned
operators. When the TVM compiler compiles these operators, it will query this
log file to get the best knob parameters.

"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado mxnet requests "Pillow<7" cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.
import sys
import os
from time import time
import torch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.contrib.download import download_testdata
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import mxnet

import vta
from vta.testing import simulator
from vta.top import graph_pack
import logging

def bring_network(ic,oc,is_,ks,st):
##############################################################################
# Download yolo net configure file, weight file, darknet library file based on
# Model Name
# ----------------------------------------------------------------------------
    class conv(nn.Module):
        def __init__(self,ic,oc,ks,st):
            super(conv, self).__init__()
            self.layer = nn.Conv2d(in_channels=ic,out_channels=oc,kernel_size=ks,stride=st,bias=False)

        def forward(self,x):
            x = self.layer(x)
            return x

    model = conv(ic,oc,ks,st)
    input_shape = [1, ic, is_, is_]
    input_data = torch.randn(input_shape)*10
    scripted_model = torch.jit.trace(model, input_data).eval()

    return scripted_model, input_shape

#logging.getLogger('autotvm').setLevel(logging.DEBUG)

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model

def compile_network(env, target, model,ic,oc,is_,ks,st):

    # Get off the shelf gluon model, and convert to relay
    net ,ishape= bring_network(ic,oc,is_,ks,st)
    mod, params = relay.frontend.from_pytorch(net, [("input0",ishape)])
    print(mod)
    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0,skip_conv_layers=None):
            mod_ = relay.quantize.quantize(mod, params=params)
    print(mod_)
    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod_["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name='cast',
            stop_name='annotation.stop_fusion',
            start_name_idx=3,
            stop_name_idx=9
        )
        print(relay_prog)
        return relay_prog, params
    else:
        return mod, params


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses an RPC session to communicate with Pynq boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up tuning, TVM uses an RPC Tracker to manage multiple devices.
# The RPC Tracker is a centralized controller node. We can register all devices to
# the tracker. For example, if we have 10 Pynq boards, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is:
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register devices to RPC Tracker
# -----------------------------------
# Now we can register our devices to the tracker. The first step is to
# build the TVM runtime for the Pynq devices.
#
# Follow :ref:`vta-index`
# to build the TVM runtime on the device. Then register the device to the tracker with:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pynq
#
# (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# After registering devices, we can confirm it by querying the rpc_tracker:
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 6 Pynq boards and 11 Raspberry Pi 3B,
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    pynq         6      6     0
#    rpi3b        11     11    0
#    ----------------------------------
#
# You can register multiple devices to the tracker to accelerate tuning.

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations.
# Here we use an Pynq-Z1 board as an example.

# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", "10.201.135.166")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 8104))

device_host = os.environ.get("VTA_RPC_HOST", "115.145.209.238")
device_port = os.environ.get("VTA_RPC_PORT", "3030")
# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
#device = "arm_cpu"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "normal_conv2d_32"

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping`
#   to larger values, makes the tuning run for longer.
#   If your device is under-powered or your conv2d operators are large, consider
#   setting a longer timeout.
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.
#
# Given that the tuning will be done on Pynq FPGA boards, make sure that
# the ```TARGET`` entry in the ``vta_config.json`` file is set to ``pynq``.


# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb_knob",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        print(tsk)
        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]

    @autotvm.template("group_conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        #print("> conv2d_packed.vta:", args)
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.group_conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, env.WGT_WIDTH)
            res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
            res = topi.cast(res, env.out_dtype)

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_group_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate():
    # Perform task extraction on Relay program
    print("Extract tasks...")
    ic,oc,is_,ks,st = 16, 32, 16, 3, 1
    relay_prog, params = compile_network(env, target, network,ic,oc,is_,ks,st)

    register_vta_tuning_tasks()

    mod = tvm.IRModule.from_expr(relay_prog)
    print("Target_host:",env.target_host)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        target=target,
        target_host=env.target_host,
    )

    # filter out non-packed conv2d task

    # We should have extracted 10 convolution tasks
    print("Extracted {} tasks:".format(len(tasks)))
    print(tasks)
    log_filename = "%s_IS_%s.log" % (device, oc)
    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    #return
    tuning_option = {
        "log_filename": log_filename,
        "tuner": "xgb_knob",
        "n_trial": 1000,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                env.TARGET,
                host=tracker_host,
                port=int(tracker_port),
                number=5,
                timeout=60,
                module_loader=vta.module_loader(),
                # check_correctness=True, # TODO: re-enable when check_correctness works again.
            ),
        ),
    }
 
    # run tuning tasks
    print("Tuning...")
    #start = time()    
    #tune_tasks(tasks, **tuning_option)
    #print("> tuning time =",time()-start)
    # evaluate with tuning history
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
    
    # compile kernels with history best records
    with autotvm.tophub.context(target):
    #with autotvm.tophub.context(target, extra_files=[log_filename]):
        # Compile network
        print("Compile to ",target.device_name,"...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=2, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )
        else:
            with vta.build_config(disabled_pass={"AlterOpLayout","tir.CommonSubexprElimTIR"}):
                graph, lib , params= relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
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

# Run the tuning and evaluate the results
tune_and_evaluate()
