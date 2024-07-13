
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
        CUDAExtension('st_spix_original_cuda', [
            # -- apis --
            'st_spix/csrc/bass/original_params.cu',
            # -- share --
            'st_spix/csrc/bass/share/gpu_utils.cu',
            "st_spix/csrc/bass/share/utils.cpp",
            # -- original --
            'st_spix/csrc/bass/original/Superpixels.cpp',
            "st_spix/csrc/bass/original/RgbLab.cu",
            "st_spix/csrc/bass/original/init_seg.cu",
            "st_spix/csrc/bass/original/sp_helper.cu",
            "st_spix/csrc/bass/original/update_param.cu",
            "st_spix/csrc/bass/original/update_seg.cu",
            "st_spix/csrc/bass/original/s_m.cu",
            # -- pybind --
            "st_spix/csrc/pybind_original.cpp",
        ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-O2','-w']}),
        CUDAExtension('st_spix_cuda', [
            # -- pairwise distance --
            'st_spix/csrc/pwd/pair_wise_distance_cuda_source.cu',
            # -- apis --
            # 'st_spix/csrc/bass/dev.cu',
            'st_spix/csrc/bass/core_params.cu',
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
            # -- flow utils --
            "st_spix/csrc/flow_utils/scatter_img.cu",
            "st_spix/csrc/flow_utils/scatter_spix.cu",
            # -- prop bass spix  --
            "st_spix/csrc/prop_spix/dev.cu",
            # -- pybind --
            "st_spix/csrc/pybind.cpp",
        ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-O2','-w']})

    ],
    cmdclass={'build_ext': BuildExtension},
)
