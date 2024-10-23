# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

import glob
import os

import torch
import torch_musa
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CppExtension
from torch_musa.utils.musa_extension import MUSAExtension
from torch_musa.utils.musa_extension import BuildExtension

requirements = ["torch", "torchvision"]

def get_musa_extensions():
    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "maskrcnn_benchmark", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_musa = glob.glob(os.path.join(extensions_dir, "musa", "*.mu"))

    sources = main_file
    extension = CppExtension

    define_macros = []

    if torch.musa.is_available():
        extension = MUSAExtension
        sources += source_musa
        define_macros += [("WITH_MUSA", None)]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]
    include_dirs.append("/home/torch_musa/")

    ext_modules = [
        extension(
            "maskrcnn_benchmark._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
    ]

    return ext_modules
    
setup(
    name="maskrcnn_benchmark",
    version="0.1",
    author="fmassa",
    url="https://github.com/facebookresearch/maskrcnn-benchmark",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_musa_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
