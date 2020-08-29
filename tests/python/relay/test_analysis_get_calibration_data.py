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

import numpy as np
import gluoncv as gcv
from PIL import Image

import tvm
import tvm.relay.testing
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay import transform
from tvm.relay.analysis import get_calibration_data
from tvm.relay.backend import compile_engine

def count_num_op(func):
    return 0

def check_data_size(mod, data, save_internal=False):
    assert len(data) == len(mod.functions) - 1
    for key, value in mod.functions.items():
        if key.name_hint != "main":
            assert len(data[key.name_hint]["inputs"]) == len(value.params)
            if save_internal:
                pass
            else:
                if isinstance(value.body, relay.Tuple):
                    assert len(data[key.name_hint]["outputs"]) == len(value.body.fields)
                else:
                    assert len(data[key.name_hint]["outputs"]) == 1

def test_simple_graph():
    # A module with two subgraphs
    mod = tvm.IRModule()

    x0 = relay.var('x0', shape=(8, 8))
    y0 = relay.var('y0', shape=(8, 8))
    z0 = x0 + y0
    z1 = x0 - y0
    z2 = relay.Tuple((z0, z1))
    f0 = relay.Function([x0, y0], z2)
    f0 = f0.with_attr("Compiler", "test_graph")
    g0 = relay.GlobalVar("g0")
    mod[g0] = f0

    x1 = relay.var('x1', shape=(8, 8))
    y1 = relay.var('y1', shape=(8, 8))
    z1 = x1 - y1
    f1 = relay.Function([x1, y1], z1)
    f1 = f1.with_attr("Compiler", "test_graph")
    g1 = relay.GlobalVar("g1")
    mod[g1] = f1

    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))
    z = relay.var('z', shape=(8, 8))
    c0 = relay.Call(g0, [x, y])
    c1 = relay.Call(g1, [relay.TupleGetItem(c0, 0), z])
    fm = relay.Function([x, y, z], c1)
    mod["main"] = fm

    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    z_data = np.random.rand(8, 8).astype('float32')
    data = get_calibration_data(mod, {"x": x_data, "y": y_data, "z": z_data})

    # Check the number and orders
    check_data_size(mod, data)
    tvm.testing.assert_allclose(data["g0"]["inputs"][0].asnumpy(), x_data)
    tvm.testing.assert_allclose(data["g0"]["inputs"][1].asnumpy(), y_data)
    tvm.testing.assert_allclose(data["g0"]["outputs"][0].asnumpy(), x_data + y_data)
    tvm.testing.assert_allclose(data["g0"]["outputs"][1].asnumpy(), x_data - y_data)
    tvm.testing.assert_allclose(data["g1"]["inputs"][0].asnumpy(), x_data + y_data)
    tvm.testing.assert_allclose(data["g1"]["inputs"][1].asnumpy(), z_data)
    tvm.testing.assert_allclose(data["g1"]["outputs"][0].asnumpy(), x_data + y_data - z_data)

def test_collect_internal_tensors():
    mod = tvm.IRModule()

    x0 = relay.var('x0', shape=(8, 8))
    y0 = relay.var('y0', shape=(8, 8))
    z0 = relay.add(x0, x0)
    z1 = relay.add(z0, y0)
    z2 = z1 - x0
    f0 = relay.Function([x0, y0], z2)
    f0 = f0.with_attr("Compiler", "test_graph")
    g0 = relay.GlobalVar("g0")
    mod[g0] = f0

    x1 = relay.var('x1', shape=(8, 8))
    y1 = relay.var('y1', shape=(8, 8))
    z1 = relay.add(x1, y1)
    f1 = relay.Function([x1, y1], z1)
    f1 = f1.with_attr("Compiler", "test_graph")
    g1 = relay.GlobalVar("g1")
    mod[g1] = f1

    x = relay.var('x', shape=(8, 8))
    y = relay.var('y', shape=(8, 8))
    z = relay.var('z', shape=(8, 8))
    c0 = relay.Call(g0, [x, y])
    c1 = relay.Call(g1, [c0, z])
    fm = relay.Function([x, y, z], c1)
    mod["main"] = fm

    x_data = np.random.rand(8, 8).astype('float32')
    y_data = np.random.rand(8, 8).astype('float32')
    z_data = np.random.rand(8, 8).astype('float32')
    data = get_calibration_data(mod, {"x": x_data, "y": y_data, "z": z_data}, True)

    # Check the number and orders
    """
    check_data_size(mod, data)
    """
    tvm.testing.assert_allclose(data["g0"]["inputs"][0].asnumpy(), x_data)
    tvm.testing.assert_allclose(data["g0"]["inputs"][1].asnumpy(), y_data)
    tvm.testing.assert_allclose(data["g0"]["outputs"][0].asnumpy(), x_data + x_data)
    tvm.testing.assert_allclose(data["g0"]["outputs"][1].asnumpy(), x_data + x_data + y_data)
    tvm.testing.assert_allclose(data["g0"]["outputs"][2].asnumpy(), x_data + y_data)
    tvm.testing.assert_allclose(data["g1"]["inputs"][0].asnumpy(), x_data + y_data)
    tvm.testing.assert_allclose(data["g1"]["inputs"][1].asnumpy(), z_data)
    tvm.testing.assert_allclose(data["g1"]["outputs"][0].asnumpy(), x_data + y_data + z_data)
    tvm.testing.assert_allclose(data["g1"]["outputs"][1].asnumpy(), x_data + y_data + z_data)

def test_mobilenet_dnnl():
    if not tvm.get_global_func("relay.ext.dnnl", True):
        print("skip because DNNL codegen is not available")
        return

    # get the model from modelzoo
    dtype = 'float32'
    ishape = (1, 3, 224, 224)
    net = gcv.model_zoo.get_model("mobilenet1.0", pretrained=True)
    mod, params = relay.frontend.from_mxnet(net, shape={'data': ishape})

    # preprocess the model
    def annotate(mod):
        func = mod["main"]

        bind_dict = {}
        for arg in func.params:
            name = arg.name_hint
            if name in params:
                bind_dict[arg] = relay.const(params[name])

        func = relay.bind(func, bind_dict)

        mod = tvm.IRModule()
        mod["main"] = func

        composite_partition = tvm.transform.Sequential([
            transform.FoldConstant(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph()
        ])

        return composite_partition(mod)

    mod = annotate(mod)

    # prepare the input image
    img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    img_path = download_testdata(img_url, 'cat.png', module='data')
    img_path = download_testdata(img_url, 'cat.png', module='data')
    image = Image.open(img_path).resize((224, 224))

    def transform_image(image):
        image = np.array(image)
        image = image - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image


    i_data = transform_image(image)

    # get the calibration data
    data = get_calibration_data(mod, {"data": i_data, **params}, True)

    # generate the code of the network with the calibration data
    target="llvm -mcpu=skylake-avx512"
    ctx = tvm.cpu()

    compile_engine.get().clear()
    with tvm.transform.PassContext(opt_level=3, config={"relay.calibration_data": {"data": data}}):
        json, lib, param = relay.build(mod, target=target, params=params)
    mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

    # execute the network
    mod.set_input("data", i_data)
    mod.run()
    out = tvm.nd.empty((1, 1000), ctx=ctx)
    out = mod.get_output(0, out)
    result = np.argmax(out.asnumpy())

    assert result == 281

if __name__ == "__main__":
    test_simple_graph()
    test_mobilenet_dnnl()
    test_collect_internal_tensors()
