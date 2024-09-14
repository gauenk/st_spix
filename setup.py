
import os

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"
# pyversion = os.environ['PYENV_VERSION']
# pyversion = pyversion if pyversion != "" else "default"

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="st_spix",
    py_modules=["st_spix"],
    install_requires=[],
    package_dir={"": "."},
    packages=find_packages("."),
    package_data={'': ['*.so']},
    include_package_data=True,
    ext_modules=[
        # CUDAExtension('st_spix_original_cuda', [
        #     # -- apis --
        #     'st_spix/csrc/bass/original_params.cu',
        #     # -- share --
        #     'st_spix/csrc/bass/share/gpu_utils.cu',
        #     "st_spix/csrc/bass/share/utils.cpp",
        #     # -- original --
        #     'st_spix/csrc/bass/original/Superpixels.cpp',
        #     "st_spix/csrc/bass/original/RgbLab.cu",
        #     "st_spix/csrc/bass/original/init_seg.cu",
        #     "st_spix/csrc/bass/original/sp_helper.cu",
        #     "st_spix/csrc/bass/original/update_param.cu",
        #     "st_spix/csrc/bass/original/update_seg.cu",
        #     "st_spix/csrc/bass/original/s_m.cu",
        #     # -- shared utils --
        #     "st_spix/csrc/bass/relabel.cu",
        #     # -- pybind --
        #     "st_spix/csrc/pybind_original.cpp",
        # ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-O1','-w']}),
        # -- keep me --
        CUDAExtension('bass_cuda', [
            # -- pairwise distance --
            'st_spix/csrc/pwd/pair_wise_distance_cuda_source.cu',
            # -- shared utils --
            "st_spix/csrc/bass/relabel.cu",
            "st_spix/csrc/prop/sparams_io.cu",
            # -- apis --
            # 'st_spix/csrc/bass/dev.cu',
            'st_spix/csrc/bass/core_params.cu',
            # "st_spix/csrc/spix_prop/split_disconnected.cu",
            # -- share --
            'st_spix/csrc/bass/share/gpu_utils.cu',
            "st_spix/csrc/bass/share/utils.cpp",
            # -- core --
            'st_spix/csrc/bass/core/Superpixels.cpp',
            "st_spix/csrc/bass/core/RgbLab.cu",
            "st_spix/csrc/bass/core/init_seg.cu",
            "st_spix/csrc/bass/core/sp_helper.cu",
            "st_spix/csrc/bass/core/update_param.cu",
            "st_spix/csrc/bass/core/update_seg.cu",
            "st_spix/csrc/bass/core/s_m.cu",
            "st_spix/csrc/bass/core/update_prop_param.cu",
            # -- flow utils --
            "st_spix/csrc/flow_utils/scatter_img.cu",
            "st_spix/csrc/flow_utils/scatter_spix.cu",
            # -- spix pooling --
            "st_spix/csrc/spix_prop/sp_pooling.cu",
            # -- pybind --
            "st_spix/csrc/pybind.cpp",
        ],extra_compile_args={'cxx': ['-g','-w',"-O0"],'nvcc': ['-w','-O0']}),
        # CUDAExtension('prop_cuda', [
        #     # -- share --
        #     'st_spix/csrc/bass/share/gpu_utils.cu',
        #     "st_spix/csrc/bass/share/utils.cpp",
        #     # -- shared utils --
        #     "st_spix/csrc/bass/relabel.cu",
        #     "st_spix/csrc/bass/sparams_io.cu",
        #     # -- prop bass spix  --
        #     "st_spix/csrc/spix_prop/dev.cu",
        #     # "st_spix/csrc/spix_prop/init_prop_seg.cu",
        #     # "st_spix/csrc/spix_prop/init_prop_seg_space.cu",
        #     "st_spix/csrc/spix_prop/fill.cu",
        #     "st_spix/csrc/spix_prop/get_params.cu",
        #     # -- modified bass updates using previous frame's prior --
        #     # "st_spix/csrc/spix_prop/update_seg_helper.cu",
        #     "st_spix/csrc/spix_prop/calc_prop_seg.cu",
        #     "st_spix/csrc/spix_prop/update_prop_seg.cu",
        #     "st_spix/csrc/spix_prop/split_disconnected.cu",
        #     # -- core --
        #     'st_spix/csrc/bass/core/Superpixels.cpp',
        #     "st_spix/csrc/bass/core/RgbLab.cu",
        #     "st_spix/csrc/bass/core/init_seg.cu",
        #     "st_spix/csrc/bass/core/sp_helper.cu",
        #     "st_spix/csrc/bass/core/update_param.cu",
        #     "st_spix/csrc/bass/core/update_seg.cu",
        #     "st_spix/csrc/bass/core/s_m.cu",
        #     "st_spix/csrc/bass/core/update_prop_param.cu",
        #     # -- pybind --
        #     "st_spix/csrc/pybind_dev.cpp",
        # ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-w']})
        CUDAExtension('prop_cuda', [
            # -- shared utils --
            "st_spix/csrc/bass/relabel.cu",
            "st_spix/csrc/prop/sparams_io.cu",
            # -- prop bass spix  --
            "st_spix/csrc/prop/fill_missing.cu",
            "st_spix/csrc/prop/split_disconnected.cu",
            "st_spix/csrc/prop/refine_missing.cu",
            # "st_spix/csrc/prop/bass_iters.cu",
            "st_spix/csrc/prop/update_prop_params.cu",
            "st_spix/csrc/prop/update_missing_seg.cu",
            "st_spix/csrc/prop/update_prop_seg.cu",
            "st_spix/csrc/prop/seg_utils.cu",
            "st_spix/csrc/prop/init_utils.cu",
            "st_spix/csrc/prop/rgb2lab.cu",
            # -- pybind --
            "st_spix/csrc/pybind_prop.cpp",
        ],extra_compile_args={'cxx': ['-g','-w',"-O0"],'nvcc': ['-w','-O0']})
    ],
    cmdclass={'build_ext': BuildExtension},
)
