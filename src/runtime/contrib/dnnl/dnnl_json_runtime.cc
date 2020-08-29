/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>
#include <cmath>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    // We need to setup constants first to get the weights scales.
    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);

    // Build the network
    BuildEngine();
  }


  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      write_to_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }
  }

 private:

  Array<NDArray> new_consts_;
  std::vector<float> weights_scales_;
  size_t conv_id_{0};

  // Build up the engine based on the input graph.
  // Return a list of constant index that should be quantized.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    // build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else {
          LOG(FATAL) << "unsupported op: " << op_name;
        }
      }
    }
  }

  // Get the minimum and maximum value of a given tensor.
  std::tuple<float, float> CollectMinMax(const NDArray tensor) {
    float* data = (float*)tensor->data;
    size_t size = 1;
    for (size_t i = 0; i < (size_t)tensor->ndim; i++) {
      size *= tensor->shape[i];
    }
    float min = data[0];
    float max = data[0];
    for (size_t i = 1; i < size; i++) {
      min = (min < data[i]) ? min : data[i];
      max = (max > data[i]) ? max : data[i];
    }
    return std::make_tuple(min, max);
  }

  // Perform quantization on constant weights by building a network
  // with only a single quantization (reorder) layer.
  int8_t* QuantizeConstWeights(const size_t& nid, 
      const DLTensor* arr, 
      const size_t& arr_size, 
      const float& scale) {
    // Initialize the network
    auto engine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    auto stream = dnnl::stream(engine);

    // Get needed information from the JSON node.
    auto node = nodes_[nid];
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    dnnl::memory::dim OC = weight_shape[0],     // output channels
        IC = input_shape[1],                    // input channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3];                   // weight width
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }

    // Preparing the DNNL memories.
    auto tag = (groups > 1) ? tag::goihw : tag::oihw;
    auto fp_weights_md = dnnl::memory::desc(weights_dims, dt::f32, tag);
    auto qt_weights_md = dnnl::memory::desc(weights_dims, dt::s8, tag);
    auto fp_mem = dnnl::memory(fp_weights_md, engine);
    auto qt_mem = dnnl::memory(qt_weights_md, engine);

    // Build the reorder layer.
    const std::vector<float> scales = {scale};
    dnnl::primitive_attr attr;
    attr.set_output_scales(0, scales);
    auto reorder_pd = dnnl::reorder::primitive_desc(engine, fp_weights_md, engine, qt_weights_md, attr);
    auto reorder_prim = dnnl::reorder(reorder_pd);

    std::unordered_map<int, dnnl::memory> reorder_args;
    reorder_args.insert({DNNL_ARG_SRC, fp_mem});
    reorder_args.insert({DNNL_ARG_DST, qt_mem});

    // Execute the reorder layer.
    write_to_dnnl_memory(arr->data, fp_mem, arr_size*4, 0);

    reorder_prim.execute(stream, reorder_args);
    stream.wait();

    int8_t* new_data = new int8_t[arr_size];
    read_from_dnnl_memory(new_data, qt_mem, arr_size, 0);

    return new_data;
  }

  // Setup the constants and quantize the weights.
  void SetupConstants(const Array<NDArray>& consts) {
    std::vector<size_t> quantized_consts;
    std::vector<size_t> weights_nids;
    size_t const_id = 0;
    // Identify the weights to be quantized by counting the number of parameters.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          if (node.HasAttr("src_scale")) {
            quantized_consts.push_back(const_id);
            weights_nids.push_back(nid);
          }
          const_id += 1;
        } else if ("dnnl.conv2d_relu" == op_name) {
          const_id += 1;
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          const_id += 2;
        } else if ("nn.dense" == op_name) {
          const_id += 1;
        } else if ("nn.batch_norm" == op_name) {
          const_id += 4;
        }
      }
    }
    // For each constant, check if we need to quantized it according to the
    // information we just gathered.
    size_t search_index = 0;
    bool complete_search = (search_index == quantized_consts.size());
    for (size_t i = 0; i < consts.size(); ++i) {
      bool need_quantized = false;
      if (!complete_search) {
        if (i == quantized_consts[search_index]) {
          need_quantized = true;
          search_index++;
        }
        complete_search = (search_index == quantized_consts.size());
      }
      if (need_quantized) {
        // Prepare the new array with new data type (int8).
        auto old_arr = consts[i];
        auto new_arr = NDArray::Empty(old_arr.Shape(), DLDataType{kDLInt, 8, 1}, {kDLCPU, 0});

        // Calculate the array size.
        size_t arr_size = 1;
        for (size_t j = 0; j < (size_t)old_arr->ndim; j++) {
          arr_size *= old_arr->shape[j];
        }

        // Gather the maximum value and calculate the scale.
        auto min_max = CollectMinMax(old_arr);
        float max_val = std::get<1>(min_max);
        float scale = 127. / max_val;
        weights_scales_.push_back(scale);

        // Get the quantized values by running the quantization.
        int8_t* new_data = QuantizeConstWeights(weights_nids[search_index-1], old_arr.operator->(), arr_size, scale);
        new_arr.CopyFromBytes(new_data, arr_size);

        // Need to delete the allocated memory.
        delete [] new_data;

        data_entry_[EntryID(const_idx_[i], 0)] = new_arr.operator->();

        // We need to keep the new array otherwise it'll be freed.
        new_consts_.push_back(new_arr);
      } else {
        data_entry_[EntryID(const_idx_[i], 0)] = consts[i].operator->();
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    CHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    CHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  dnnl::primitive BuildReorderPrimitive(dnnl::memory src_memory, dnnl::memory dst_memory, const int mask, float scale) {
    const std::vector<float> scales = {scale};
    dnnl::primitive_attr attr;
    attr.set_output_scales(mask, scales);
    auto reorder_pd = dnnl::reorder::primitive_desc(engine_, src_memory.get_desc(), engine_, dst_memory.get_desc(), attr);
    return dnnl::reorder(reorder_pd);
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false,
              const bool quantize = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    // For quantized operations.
    bool is_quantized = false;
    bool is_dequantized = false;
    float src_scale = 0.0f;
    float weights_scale = 0.0f;
    float bias_scale = 0.0f;
    float dst_scale = 0.0f;
    // Get the quantization parameters from the node.
    if (node.HasAttr("is_quantized")) {
      std::vector<std::string> src_attrs = node.GetAttr<std::vector<std::string>>("src_scale");
      std::vector<std::string> quantize_attrs = node.GetAttr<std::vector<std::string>>("is_quantized");
      std::vector<std::string> dequantize_attrs = node.GetAttr<std::vector<std::string>>("is_dequantized");
      src_scale = std::stof(src_attrs[0]);
      weights_scale = weights_scales_[conv_id_];  // weight scale is gathered from SetUpConstants
      bias_scale = src_scale * weights_scale;
      dst_scale = 1.0f / bias_scale;
      is_quantized = std::stoi(quantize_attrs[0]);
      is_dequantized = std::stoi(dequantize_attrs[0]);
      conv_id_++;
    }

    dnnl::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    // Select the right data type according to whether the layer is quantized.
    auto conv_src_md = dnnl::memory::desc(src_dims, is_quantized ? dt::u8 : dt::f32 , tag::any);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, is_quantized ? dt::s8 : dt::f32, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, is_quantized ? dt::s32 : dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, is_quantized ? dt::s32 : dt::f32, tag::nchw);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    dnnl::primitive_attr attr;
    // Enable ReLU
    if (has_relu) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }


    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Data memory.
    CHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::f32, tag::nchw});

    // Weight memory.
    CHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, {weights_dims, is_quantized ? dt::s8 : dt::f32, (groups > 1) ? tag::goihw : tag::oihw});

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory({bias_dims, is_quantized ? dt::s32 : dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_dnnl_memory(bias, conv2d_bias_memory, is_quantized ? OC * sizeof(int) : OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, {{dst_dims}, dt::f32, tag::nchw});


    // Push to the network and bind memory buffers.
    if (is_quantized) {
      // Prepare the quantized memories
      auto quantized_src_memory = dnnl::memory(conv2d_prim_desc.src_desc(), engine_);
      auto quantized_dst_memory = dnnl::memory(conv2d_prim_desc.dst_desc(), engine_);

      net_.push_back(BuildReorderPrimitive(conv2d_src_memory, quantized_src_memory, 0, src_scale));
      net_args_.push_back({{DNNL_ARG_FROM, conv2d_src_memory}, {DNNL_ARG_TO, quantized_src_memory}});

      net_.push_back(dnnl::convolution_forward(conv2d_prim_desc));

      auto src_memory = quantized_src_memory;
      auto weights_memory = conv2d_weights_memory;
      auto dst_memory = is_dequantized ? quantized_dst_memory : conv2d_dst_memory;
      net_args_.push_back({{DNNL_ARG_SRC, src_memory},
                           {DNNL_ARG_WEIGHTS, weights_memory},
                           {DNNL_ARG_BIAS, conv2d_bias_memory},
                           {DNNL_ARG_DST, dst_memory}});

      if (is_dequantized) {
        net_.push_back(BuildReorderPrimitive(quantized_dst_memory, conv2d_dst_memory, 0, dst_scale));
        net_args_.push_back({{DNNL_ARG_FROM, quantized_dst_memory}, {DNNL_ARG_TO, conv2d_dst_memory}});
      }
    } else {
      net_.push_back(dnnl::convolution_forward(conv2d_prim_desc));

      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                           {DNNL_ARG_BIAS, conv2d_bias_memory},
                           {DNNL_ARG_DST, conv2d_dst_memory}});
    }
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    float bias[OC] = {0};
    write_to_dnnl_memory(bias, bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_md = dnnl::memory::desc{{shape}, dt::f32, tag::abcd};

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    CHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto out_md = dnnl::memory::desc(shape, dt::f32, tag::abcd);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    CHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    CHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto add_desc =
        dnnl::binary::desc(dnnl::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = dnnl::binary::primitive_desc(add_desc, engine_);
    auto add = dnnl::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
