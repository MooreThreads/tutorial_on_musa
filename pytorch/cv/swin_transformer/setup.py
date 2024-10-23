import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
if os.getenv('FORCE_CUDA', '0') == '1':
    setup(name='swin_window_process',
        ext_modules=[
            CUDAExtension('swin_window_process', [
                'swin_window_process.cpp',
                'swin_window_process_kernel.cu',
            ])
        ],
        cmdclass={'build_ext': BuildExtension})
elif os.getenv('FORCE_MUSA', '0') == '1':
    from torch_musa.utils.simple_porting import SimplePorting
    from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension
    SimplePorting(cuda_dir_path="../window_process",
                  mapping_rule={
                    "kCUDA": "kPrivateUse1",
                    "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",
                    }).run()
    setup(name='swin_window_process',
        ext_modules=[
            MUSAExtension('swin_window_process', [
                '../window_process_musa/swin_window_process.cpp',
                '../window_process_musa/swin_window_process_kernel.mu',
            ])
        ],
        cmdclass={'build_ext': BuildExtension})

