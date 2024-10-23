// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once

#ifdef WITH_MUSA
#include "musa/vision.h"
#endif


std::tuple<at::Tensor, at::Tensor> ROIPool_forward(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width) {
  if (input.is_privateuseone()) {
#ifdef WITH_MUSA
    return ROIPool_forward_musa(input, rois, spatial_scale, pooled_height, pooled_width);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor ROIPool_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const int batch_size,
                                 const int channels,
                                 const int height,
                                 const int width) {
  if (grad.is_privateuseone()) {
#ifdef WITH_MUSA
    return ROIPool_backward_musa(grad, input, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width);
#else
    AT_ERROR("Not compiled with MUSA support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}



