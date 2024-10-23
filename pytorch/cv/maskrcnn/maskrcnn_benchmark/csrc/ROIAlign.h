// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#ifdef WITH_MUSA
#include "musa/vision.h"
#endif

// Interface for Python
at::Tensor ROIAlign_forward(const at::Tensor& input,
                            const at::Tensor& rois,
                            const float spatial_scale,
                            const int pooled_height,
                            const int pooled_width,
                            const int sampling_ratio) {
  if (input.is_privateuseone()) {
#ifdef WITH_MUSA
    return ROIAlign_forward_musa(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ROIAlign_backward(const at::Tensor& grad,
                             const at::Tensor& rois,
                             const float spatial_scale,
                             const int pooled_height,
                             const int pooled_width,
                             const int batch_size,
                             const int channels,
                             const int height,
                             const int width,
                             const int sampling_ratio) {
  if (grad.is_privateuseone()) {
#ifdef WITH_MUSA
    return ROIAlign_backward_musa(grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

