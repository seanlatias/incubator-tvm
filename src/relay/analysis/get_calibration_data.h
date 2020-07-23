#include <tvm/runtime/registry.h>

namespace tvm {

struct CalibrationDataNode : public AttrsNode<CalibrationDataNode> {
  Map<String, Map<String, Array<runtime::NDArray>>> data;

  TVM_DECLARE_ATTRS(CalibrationDataNode, "relay.analysis.CalibrationData") {
    TVM_ATTR_FIELD(data)
      .describe("Calibration data.")
      .set_default(NullValue<Map<String, Map<String, Array<runtime::NDArray>>>>());
  }
};

class CalibrationData : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CalibrationData, Attrs, CalibrationDataNode);
};

TVM_REGISTER_NODE_TYPE(CalibrationDataNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.calibration_data", CalibrationData);

}  // namepsace tvm 
