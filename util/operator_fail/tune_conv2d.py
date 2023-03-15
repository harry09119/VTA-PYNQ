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

"""Tuning a single conv2d operator"""

from collections import namedtuple
import logging
import os
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import vta
import vta.testing

env = vta.get_env()

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

resnet_wkls = [
    # Workloads of resnet18 on imagenet
    # ('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    ("resnet-18.C2", Workload(env.BATCH, 32, 32, 16, 32, 3, 3, 0, 0, 1, 1)),
    #("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    #("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    #("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    #("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    #("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    #("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    #("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    #("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    #("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
]


@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x

@autotvm.template("conv2d_packed.vta")
def conv2d(N, CI, H, W, CO, KH, KW, strides, padding, dilation):
    # 2D convolution layer dimensions taken from ResNet-18 architecture
    # (9th convolutional layer)
    with tvm.target.vta():
        batch_size = 1
        height = 32
        width = 32
        in_channels = 16
        out_channels = 32
        kernel_h = 3
        kernel_w = 3
        pad_h = 0
        pad_w = 0
        stride_h = 1
        stride_w = 1
        assert batch_size % env.BATCH == 0
        assert in_channels % env.BLOCK_IN == 0
        assert out_channels % env.BLOCK_OUT == 0

        # Input feature map: (N, IC, H, W, n, ic)
        data_shape = (
            batch_size // env.BATCH,
            in_channels // env.BLOCK_IN,
            height,
            width,
            env.BATCH,
            env.BLOCK_IN,
        )
        # Kernel: (OC, IC, H, W, oc, ic)
        kernel_shape = (
            out_channels // env.BLOCK_OUT,
            in_channels // env.BLOCK_IN,
            kernel_h,
            kernel_w,
            env.BLOCK_OUT,
            env.BLOCK_IN,
        )
        # Derive output feature map dimensions
        fout_height = (height + 2 * pad_h - kernel_h) // stride_h + 1
        fout_width = (width + 2 * pad_w - kernel_w) // stride_w + 1
        # Output feature map: (N, OC, H, W, n, oc)
        output_shape = (
            batch_size // env.BATCH,
            out_channels // env.BLOCK_OUT,
            fout_height,
            fout_width,
            env.BATCH,
            env.BLOCK_OUT,
        )

        # Convolution reduction axes
        dy = te.reduce_axis((0, kernel_h), name="dy")
        dx = te.reduce_axis((0, kernel_w), name="dx")
        ic = te.reduce_axis((0, in_channels // env.BLOCK_IN), name="ic")
        ic_tns = te.reduce_axis((0, env.BLOCK_IN), name="ic_tns")

        # Input placeholder tensors
        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
        
        # Copy buffers:
        #   Apply spatial padding to input feature map
        data_buf = topi.nn.pad(data, [0, 0, pad_h, pad_w, 0, 0], name="data_buf")
        data_buf = te.compute(data_shape, lambda *i: data_buf(*i), "data_buf")
        kernel_buf = te.compute(kernel_shape, lambda *i: kernel(*i), "kernel_buf")

        # Declare 2D convolution
        res_conv = te.compute(
            output_shape,
            lambda bo, co, i, j, bi, ci: te.sum(
                data_buf[bo, ic, i * stride_h + dy, j * stride_w + dx, bi, ic_tns].astype(env.acc_dtype)
                * kernel_buf[co, ic, dy, dx, ci, ic_tns].astype(env.acc_dtype),
                axis=[ic, dy, dx, ic_tns],
            ),
            name="res_conv",
        )
       
        # Add shift stage for fix-point normalization
        res_shr = te.compute(output_shape, lambda *i: res_conv(*i) >> 8, name="res_shr")

        # Apply clipping between (0, input max value)
        #res_min = te.compute(output_shape, lambda *i: tvm.te.min(res_shr(*i), tvm.tir.const(127,env.acc_dtype)), name="res_min")
        #res_max = te.compute(output_shape, lambda *i: tvm.te.max(res_min(*i), tvm.tir.const(0,env.acc_dtype)), name="res_max")

        # Result Tensor
        res = te.compute(output_shape, lambda *i: res_shr(*i).astype(env.acc_dtype), name="res")
        # Create TVM schedule
        s = te.create_schedule(res.op)
        # Let's look at the default TVM schedule
        #print(tvm.lower(s, [data, kernel, res], simple_mode=True))
        data_buf
        # Let's define tiling sizes
        b_, oc_, y_, x_, _, _ = s[res_conv].op.axis
        ic_, _, _, _ = s[res_conv].op.reduce_axis
        cfg = autotvm.get_config()
        cfg.define_split("tile_b", b_, num_outputs=2)
        cfg.define_split("tile_oc", oc_, num_outputs=2)
        cfg.define_split("tile_y", y_, num_outputs=2)
        cfg.define_split("tile_x", x_, num_outputs=2)
        cfg.define_split("tile_ic", ic_, num_outputs=2)
        #cfg.define_knob("oc_nthread", [1, 2])
        #cfg.define_knob("h_nthread", [1, 2])
        
        cfg.add_flop(
            2
            * np.prod(output_shape)
            * kernel_h
            * kernel_w
            * (in_channels // env.BLOCK_IN)
            * env.BLOCK_IN
        )
        #  the batch dimension has no effect)
        b, oc, y, x, b_tns, oc_tns = s[res].op.axis
        b_out, b_inn = cfg["tile_b"].apply(s, res, b)
        oc_out, oc_inn = cfg["tile_oc"].apply(s,res,oc)
        y_out, y_inn = cfg["tile_y"].apply(s,res,y)
        x_out, x_inn = cfg["tile_x"].apply(s,res,x)
        s[res].reorder(b_out, oc_out, y_out, x_out, b_inn, oc_inn, y_inn, x_inn, b_tns, oc_tns)

        # Move intermediate computation into each output compute tile
        s[res_conv].compute_at(s[res], x_out)
        s[res_shr].compute_at(s[res], x_out)
        #s[res_min].compute_at(s[res], x_out)
        #s[res_max].compute_at(s[res], x_out)

        # Apply additional loop split along reduction axis (input channel)
        b_inn, oc_inn, y_inn, x_inn, b_tns, oc_tns = s[res_conv].op.axis
        ic_out, ic_inn = cfg["tile_ic"].apply(s,res_conv,ic)

        s[res_conv].reorder(ic_out, b_inn, oc_inn, y_inn, ic_inn, dy, dx, x_inn, b_tns, oc_tns, ic_tns)
        
        _, tx = s[res].split(oc_out, factor=2)
        s[res].reorder(tx, b_out)
        s[res].bind(tx, te.thread_axis("cthread"))
        # VTA only supports 2 virtual threads
        """
        if cfg["oc_nthread"].val > 1:
            _, v_t = s[res].split(oc_out, factor=cfg["oc_nthread"].val)
            s[res].reorder(v_t, b)
            s[res].bind(v_t, te.thread_axis("cthread"))

        # virtual threading along spatial rows
        if cfg["h_nthread"].val > 1:
            _, v_t = s[res].split(y_out, factor=cfg["h_nthread"].val)
            s[res].reorder(v_t, b)
            s[res].bind(v_t, te.thread_axis("cthread"))
        """
        # Let's look at the current TVM schedule after blocking and virtual threading
        #print(tvm.lower(s, [data, kernel, res], simple_mode=True))

        # Set scope of SRAM buffers
        s[data_buf].set_scope(env.inp_scope)
        s[kernel_buf].set_scope(env.wgt_scope)
        s[res_conv].set_scope(env.acc_scope)
        s[res_shr].set_scope(env.acc_scope)
        #s[res_min].set_scope(env.acc_scope)
        #s[res_max].set_scope(env.acc_scope)

        # Block data and kernel cache reads
        s[data_buf].compute_at(s[res_conv], ic_out)
        s[kernel_buf].compute_at(s[res_conv], ic_out)

        # Use DMA copy pragma on DRAM->SRAM operations
        s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)
        s[kernel_buf].pragma(s[kernel_buf].op.axis[0], env.dma_copy)

        s[res].pragma(s[res].op.axis[4], env.dma_copy)

        # Apply tensorization over the batch tensor tile axis
        s[res_conv].tensorize(b_tns, env.gemm)

        # Add an ALU pragma over the shift and clipping operations
        s[res_shr].pragma(s[res_shr].op.axis[0], env.alu)
        #s[res_min].pragma(s[res_min].op.axis[0], env.alu)
        #s[res_max].pragma(s[res_max].op.axis[0], env.alu)
    return s, [data,kernel, res]
"""
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI // env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N // env.BATCH, CO // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

    with tvm.target.vta():
        res = topi.nn.conv2d(
            input=data,
            filter=kernel,
            padding=padding,
            strides=strides,
            dilation=dilation,
            out_dtype=env.acc_dtype,
        )
        res = topi.right_shift(res, env.WGT_WIDTH)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)

    if tvm.target.Target.current().device_name == "vta":
        s = topi.generic.schedule_conv2d_nchw([res])
    else:
        s = te.create_schedule([res.op])
    return s, [data, kernel, bias, res]
"""
if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)
    logging.getLogger('autotvm').setLevel(logging.DEBUG)
    # Tuning log files
    log_file = "%s.conv2d.log" % (env.TARGET)
    # create tmp log file
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Get tracker info from env
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "10.201.135.166")
    tracker_port = os.environ.get("TVM_TRACKER_PORT", 8104)
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    for idx, (wl_name, wl) in enumerate(resnet_wkls):
        prefix = "[Task %2d/%2d] " % (idx, len(resnet_wkls))

        # Read in workload parameters
        N = wl.batch
        CI = wl.in_filter
        H = wl.height
        W = wl.width
        CO = wl.out_filter
        KH = wl.hkernel
        KW = wl.wkernel
        strides = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        dilation = (1, 1)

        # Create task
        task = autotvm.task.create(
            "conv2d_packed.vta",
            args=(N, CI, H, W, CO, KH, KW, strides, padding, dilation),
            target=tvm.target.vta(),
            target_host=env.target_host,
        )
        #print(task.config_space)

        # Tune
        measure_option = autotvm.measure_option(
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
        )

        # Run Tuner
        tuner = autotvm.tuner.XGBTuner(task, loss_type="rank", feature_type="knob")
        tuner.tune(
            n_trial=min(1000, len(task.config_space)),
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(len(task.config_space), prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # Pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)
