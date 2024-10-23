#include <torch/script.h>
#include <torch_musa/csrc/core/Device.h>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, const char* argv[]) {
  // Register 'musa' for PrivateUse1 as we save model with 'musa'.
  c10::register_privateuse1_backend("musa");

  torch::jit::script::Module module;
  try {
    // Load model which saved with torch jit.trace or jit.script.
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::vector<torch::jit::IValue> inputs;
  // Ready for input data.
  torch::Tensor input = torch::rand({1, 3, 224, 224}).to("musa");
  inputs.push_back(input);

  // Model execute.
  auto output_ivalue = module.forward(inputs);
  if (output_ivalue.isTuple()) {
    auto output_tuple = output_ivalue.toTuple();
    std::cout << "Output is a tuple with " << output_tuple->elements().size() << " elements." << std::endl;
    for (size_t i = 0; i < output_tuple->elements().size(); ++i) {
      std::cout << "Output element " << i << ":" << std::endl;
      auto& element = output_tuple->elements()[i];
      if (element.isTensor()) {
        auto out_tensor = element.toTensor();
        std::cout << "dim 0: " << out_tensor.sizes()[0] << std::endl;
        std::cout << "dim 1: " << out_tensor.sizes()[1] << std::endl;
        std::cout << "dim 2: " << out_tensor.sizes()[2] << std::endl;
        std::cout << "dim 3: " << out_tensor.sizes()[3] << std::endl;
      } else if (element.isTensorList()) {
	      auto tensor_list = element.toTensorList();
        std::cout << "Element " << i << " is a list with " << tensor_list.size() << " elements." << std::endl;
        for (size_t j = 0; j < tensor_list.size(); ++j) {
          torch::Tensor tensor = tensor_list[i];
          std::cout << "Tensor " << j << " in list:" << std::endl;
          std::cout << "dim 0: " << tensor.size(0) << std::endl;
          std::cout << "dim 1: " << tensor.size(1) << std::endl;
          std::cout << "dim 2: " << tensor.size(2) << std::endl;
          std::cout << "dim 3: " << tensor.size(3) << std::endl;
        }
      } else {
        std::cerr << "Error: Element " << i << " is not a tensor." << std::endl;
      }
    }
  } else {
    at::Tensor output = output_ivalue.toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
  }

  return 0;
}
