#pragma once

#ifdef WITH_MUSA
#include "musa/vision.h"
#endif

// Interface for Python
at::Tensor SigmoidFocalLoss_forward(
		const at::Tensor& logits,
                const at::Tensor& targets,
		const int num_classes, 
		const float gamma, 
		const float alpha) {
  if (logits.is_privateuseone()) {
#ifdef WITH_MUSA
    return SigmoidFocalLoss_forward_musa(logits, targets, num_classes, gamma, alpha);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor SigmoidFocalLoss_backward(
			     const at::Tensor& logits,
                             const at::Tensor& targets,
			     const at::Tensor& d_losses,
			     const int num_classes,
			     const float gamma,
			     const float alpha) {
  if (logits.is_privateuseone()) {
#ifdef WITH_MUSA
    return SigmoidFocalLoss_backward_musa(logits, targets, d_losses, num_classes, gamma, alpha);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
