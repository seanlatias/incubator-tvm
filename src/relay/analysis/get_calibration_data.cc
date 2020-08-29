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
 * \file src/relay/analysis/get_calibration_data.cc
 *
 * \brief To get the calibration data, we need to perform two
 * steps. First, we need to prepare the module that generates
 * the tensor values (GetCalibrateModule). Second, we need to
 * generate the mapping between the values and the functions
 * (GetCalibrateOutputMap).
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

/*!
 * \brief This function returns a module that will be used by
 * the relay graph runtime for collecting the calibration data.
 * To do that, we first make all inputs and outputs of each
 * function into the final output (i.e., the final output is a
 * tuple of tensors). Then, we change the compiler attribute of
 * each function. Finally, we mark all function to be inlined.
 */

class Updater : public ExprRewriter {
 public:
  Updater(const IRModule& module, bool is_main, const Array<String>& save_op_list) : module_(module), is_main_(is_main), save_op_list_(save_op_list) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (is_main_) {
      if (call->op->IsInstance<GlobalVarNode>()) {
        auto var = Downcast<GlobalVar>(call->op);
        CHECK(module_->ContainGlobalVar(var->name_hint)) << "Function " << var << " is not defined";
        auto func = Downcast<Function>(module_->Lookup(var));
        if (func->GetAttr<String>(attr::kCompiler)) {
          if (auto* tn = func->body->checked_type_.as<TupleTypeNode>()) {
            size_t size = tn->fields.size();
            auto tuple_get_item = TupleGetItem(post, size-1);
            tuple_get_item->checked_type_ = call->checked_type_;
            return tuple_get_item;
          }
        }
      }
    } else {
      bool collect_output = save_op_list_.empty();
      for (const auto& name : save_op_list_) {
        if (const auto* op_node = call->op.as<OpNode>()) {
          collect_output |= (name == op_node->name);
        }
      }
      if (collect_output) {
        // special case for collectign the output of batch norm
        // we only need to return the first value in the output tuple
        if (const auto* op_node = call->op.as<OpNode>()) { 
          if (op_node->name == "nn.batch_norm") {
            auto tuple_get_item = TupleGetItem(post, 0);
            auto* tn = call->checked_type_.as<TupleTypeNode>();
            tuple_get_item->checked_type_ = tn->fields[0];
            new_outputs_.push_back(tuple_get_item);
            return post;
          }
        }
        new_outputs_.push_back(post);
      }
    }
    return post;
  }

  Array<Expr> GetNewOutputs() { return new_outputs_; }

 private:
  const IRModule& module_;
  bool is_main_;
  const Array<String>& save_op_list_;
  Array<Expr> new_outputs_;
};

class Collector : public ExprRewriter {
 public:
  explicit Collector(const IRModule& module) : module_(module) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    // check if the function implementation is available
    // intrinsic functions are excluded for now
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto var = Downcast<GlobalVar>(call->op);
      CHECK(module_->ContainGlobalVar(var->name_hint)) << "Function " << var << " is not defined";
      // we only handle functions with Compiler attribute set
      auto func = Downcast<Function>(module_->Lookup(var));
      if (func->GetAttr<String>(attr::kCompiler)) {
        // collect all the inputs and outputs
        for (const auto& it : call->args) new_outputs_.push_back(it);
        new_outputs_.push_back(post);
      }
    }
    return post;
  }

  Array<Expr> GetNewOutputs() { return new_outputs_; }

 private:
  const IRModule& module_;
  Array<Expr> new_outputs_;
};

Expr FlattenOutputTuple(const Array<Expr>& exprs) {
  Array<Expr> fields;
  for (const auto& it : exprs) {
    CHECK(it->checked_type_.defined());
    if (auto* tn = it->checked_type_.as<TupleTypeNode>()) {
      // TODO(seanlatias): for now input argument cannot be a tuple
      // CHECK(it->IsInstance<CallNode>());
      for (size_t i = 0; i < tn->fields.size(); i++) {
        fields.push_back(TupleGetItem(it, i));
      }
    } else {
      fields.push_back(it);
    }
  }
  return Tuple(fields);
}

IRModule GetCalibrateModule(IRModule module, const bool save_internal_tensor, const Array<String>& save_op_list) {
  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  auto main_var = module->GetGlobalVar("main");
  if (save_internal_tensor) { 
    // first gather the output of each op inside a function with compiler attr being set.
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        if (func->GetAttr<String>(attr::kCompiler)) {
          Updater updater(module, false, save_op_list);
          PostOrderRewrite(func->body, &updater);
          auto new_outputs = updater.GetNewOutputs();
          new_outputs.push_back(func->body);
          Expr tuple = FlattenOutputTuple(new_outputs);
          func = Function(func->params, tuple, tuple->checked_type_, func->type_params, func->attrs);
          module->Update(pair.first, func);
        }
      }
    }
    // then handle the main function
    auto main_func = Downcast<Function>(module->Lookup(main_var));
    Updater updater(module, true, save_op_list);
    auto updated = PostOrderRewrite(main_func->body, &updater);
    main_func = Function(main_func->params, updated, main_func->ret_type, main_func->type_params, main_func->attrs); 
    module->Update(main_var, main_func);
  }
  // collect the in/out of each function and set them as the output of main function.
  auto main_func = Downcast<Function>(module->Lookup(main_var));
  Collector collector(module);
  PostOrderRewrite(main_func->body, &collector);
  auto new_outputs = collector.GetNewOutputs();
  Expr tuple = FlattenOutputTuple(new_outputs);
  main_func = Function(main_func->params, tuple, tuple->checked_type_, main_func->type_params, main_func->attrs);
  module->Update(main_var, main_func);
  // reset the attribute of functions for running graph runtime
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      if (func->GetAttr<String>(attr::kCompiler)) {
        // get the updated function
        func = Downcast<Function>(module->Lookup(pair.first));
        // we need to inline the functions in order to run grpah runtime
        func = WithAttr(std::move(func), attr::kInline, tvm::Integer(1));
        // reset the compiler attribute to null for llvm execution
        func = WithAttr(std::move(func), attr::kCompiler, NullValue<ObjectRef>());
        module->Update(pair.first, func);
      }
    }
  }
  return module;
}

/*!
 * \brief This function generates the output mapping between
 * the calibration data and each function. The key is a
 * GlobalVar that corresponds to each function and the value
 * is an array of integers. The size of the array is always
 * three. The first value is the offset the points to the start.
 * The second value is the number of inputs. The third value
 * is the number of outputs.
 */

class OutputMapper : public ExprRewriter {
 public:
  OutputMapper(Map<String, Array<Integer>>* output_map, const IRModule& module, size_t* offset)
      : output_map_(output_map), module_(module), offset_(offset) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto var = Downcast<GlobalVar>(call->op);
      CHECK(module_->ContainGlobalVar(var->name_hint)) << "Function " << var << " is not defined";
      CHECK_EQ(output_map_->count(var->name_hint), 0)
          << "Repeated function call " << var << " is not supported.";
      auto func = Downcast<Function>(module_->Lookup(var));
      Array<Integer> info;
      // the first value is the offset
      info.push_back(Integer(*offset_));
      // the second value is the number of inputs
      info.push_back(Integer(call->args.size()));
      // the third value is the number of outputs
      // we need to check if the output is a tuple
      size_t out_size = 1;
      if (auto* tn = func->body.as<TupleNode>()) {
        info.push_back(Integer(tn->fields.size()));
        out_size = tn->fields.size();
      } else {
        info.push_back(Integer(1));
      }
      output_map_->Set(var->name_hint, info);
      // calculate the offset for the next function
      *offset_ = *offset_ + call->args.size() + out_size;
    }
    return post;
  }

 private:
  Map<String, Array<Integer>>* output_map_;
  const IRModule& module_;
  size_t* offset_;
};

Map<String, Array<Integer>> GetCalibrateOutputMap(const IRModule& module) {
  Map<String, Array<Integer>> output_map;
  size_t offset = 0;
  auto glob_funcs = module->functions;
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      if (pair.first->name_hint == "main") {
        OutputMapper output_mapper(&output_map, module, &offset);
        auto func = GetRef<Function>(fn);
        PostOrderRewrite(func->body, &output_mapper);
      }
    }
  }

  return output_map;
}

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_module").set_body_typed([](IRModule mod, const bool save_internal_tensor, const Array<String>& save_op_list) {
  return GetCalibrateModule(mod, save_internal_tensor, save_op_list);
});

TVM_REGISTER_GLOBAL("relay.analysis.get_calibrate_output_map")
    .set_body_typed([](const IRModule& mod) { return GetCalibrateOutputMap(mod); });

}  // namespace relay
}  // namespace tvm
