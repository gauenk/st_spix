
# -- local --
import os
# os.environ['PYTORCH_NVCC'] = "ccache nvcc"
# os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '1' # "1" # for faster
# os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '1' # "1" # for fasterb
os.environ['TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES'] = '0' # "1" # for fasterb

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
        # CUDAExtension('bass_cuda', [
        #     # -- pairwise distance --
        #     'st_spix/csrc/pwd/pair_wise_distance_cuda_source.cu',
        #     # -- shared utils --
        #     "st_spix/csrc/bass/relabel.cu",
        #     "st_spix/csrc/prop/simple_sparams_io.cu",
        #     "st_spix/csrc/prop/init_utils.cu",
        #     "st_spix/csrc/prop/simple_init_sparams.cu",
        #     # -- apis --
        #     # 'st_spix/csrc/bass/dev.cu',
        #     'st_spix/csrc/bass/core_params.cu',
        #     # "st_spix/csrc/spix_prop/split_disconnected.cu",
        #     # -- share --
        #     'st_spix/csrc/bass/share/gpu_utils.cu',
        #     "st_spix/csrc/bass/share/utils.cpp",
        #     # -- core --
        #     'st_spix/csrc/bass/core/Superpixels.cpp',
        #     "st_spix/csrc/bass/core/RgbLab.cu",
        #     "st_spix/csrc/bass/core/init_seg.cu",
        #     "st_spix/csrc/bass/core/sp_helper.cu",
        #     "st_spix/csrc/bass/core/update_param.cu",
        #     "st_spix/csrc/bass/core/update_seg.cu",
        #     "st_spix/csrc/bass/core/s_m.cu",
        #     "st_spix/csrc/bass/core/update_prop_param.cu",
        #     # -- flow utils --
        #     "st_spix/csrc/flow_utils/scatter_img.cu",
        #     "st_spix/csrc/flow_utils/scatter_spix.cu",
        #     # -- spix pooling --
        #     "st_spix/csrc/spix_prop/sp_pooling.cu",
        #     # -- pybind --
        #     "st_spix/csrc/pybind.cpp",
        # ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-w']}),




        # -- keep me --
        CUDAExtension('bist_cuda', [
            # -- shared utils --
            "st_spix/csrc/bist/pyapi.cu",
            "st_spix/csrc/bist/atomic_helpers.cu",
            # "st_spix/csrc/bist/file_io.cpp",
            "st_spix/csrc/bist/init_utils.cu",
            "st_spix/csrc/bist/init_seg.cu",
            "st_spix/csrc/bist/init_sparams.cu",
            "st_spix/csrc/bist/rgb2lab.cu",
            "st_spix/csrc/bist/compact_spix.cu",
            "st_spix/csrc/bist/seg_utils.cu",
            "st_spix/csrc/bist/update_params.cu",
            "st_spix/csrc/bist/update_seg.cu",
            "st_spix/csrc/bist/split_merge.cu",
            "st_spix/csrc/bist/split_merge_orig.cu",
            "st_spix/csrc/bist/split_merge_prop.cu",
            "st_spix/csrc/bist/sparams_io.cu",
            "st_spix/csrc/bist/shift_and_fill.cu",
            "st_spix/csrc/bist/shift_labels.cu",
            "st_spix/csrc/bist/fill_missing.cu",
            "st_spix/csrc/bist/sp_pooling.cu",
            "st_spix/csrc/bist/split_disconnected.cu",
            "st_spix/csrc/bist/relabel.cu",
            # "st_spix/csrc/bist/demo_utils.cu",
            "st_spix/csrc/bist/bass.cu",
            "st_spix/csrc/bist/prop.cu",
            # -- pybind --
            "st_spix/csrc/pybind_bist.cpp",
        ],
        libraries=['cuda', 'cublas', 'cudadevrt'],
        extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-w','--extended-lambda']}),


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






        # CUDAExtension('st_attn_cuda', [
        #     # -- superpixel attention --
        #     "st_spix/csrc/attn/sim_sum.cu",
        #     # -- pybind --
        #     "st_spix/csrc/pybind_attn.cpp",
        # ],
        # extra_compile_args={'cxx':['-g','-w'],'nvcc':['-w']},),

        # CUDAExtension('prop_cuda', [
        #     # -- shared utils --
        #     "st_spix/csrc/prop/pch.cu",
        #     "st_spix/csrc/bass/relabel.cu",
        #     "st_spix/csrc/prop/sparams_io.cu",
        #     "st_spix/csrc/prop/sp_video_pooling.cu",
        #     "st_spix/csrc/prop/simple_sparams_io.cu",
        #     "st_spix/csrc/prop/refine.cu",
        #     # -- prop utils --
        #     "st_spix/csrc/prop/rgb2lab.cu",
        #     "st_spix/csrc/prop/seg_utils.cu",
        #     "st_spix/csrc/prop/init_seg.cu",
        #     "st_spix/csrc/prop/init_utils.cu",
        #     "st_spix/csrc/prop/init_sparams.cu",
        #     "st_spix/csrc/prop/simple_init_sparams.cu",
        #     # -- tools --
        #     "st_spix/csrc/prop/shift_labels.cu",
        #     "st_spix/csrc/prop/shift_tensor.cu",
        #     "st_spix/csrc/prop/shift_order.cu",
        #     "st_spix/csrc/prop/shift_tensor_ordered.cu",
        #     "st_spix/csrc/prop/split_disconnected.cu",
        #     "st_spix/csrc/prop/fill_missing.cu",
        #     # -- standard bass --
        #     "st_spix/csrc/prop/bass.cu",
        #     "st_spix/csrc/prop/update_params.cu",
        #     "st_spix/csrc/prop/update_seg.cu",
        #     "st_spix/csrc/prop/split_merge.cu",
        #     "st_spix/csrc/prop/split_merge_orig.cu",
        #     "st_spix/csrc/prop/simple_split_merge.cu",
        #     # -- prop bass spix  --
        #     "st_spix/csrc/prop/simple_refine_missing.cu",
        #     "st_spix/csrc/prop/refine_missing.cu",
        #     "st_spix/csrc/prop/prop_bass.cu",
        #     "st_spix/csrc/prop/update_prop_params.cu",
        #     "st_spix/csrc/prop/update_missing_seg.cu",
        #     "st_spix/csrc/prop/update_prop_seg.cu",
        #     "st_spix/csrc/prop/split_merge_prop.cu",
        #     # "st_spix/csrc/prop/merge_prop.cu",
        #     # "st_spix/csrc/prop/split_prop.cu",
        #     "st_spix/csrc/prop/relabel.cu",
        #     # -- pybind --
        #     "st_spix/csrc/pybind_prop.cpp",
        # ],
        # extra_compile_args={'cxx':['-g','-w',"-O0"],
        #                     # 'nvcc':['-w','-rdc=true']},
        #                     'nvcc':["-G","-O0",'-w']},
        # )

        # 'nvcc':['-w','-G']},)
        #extra_compile_args={'cxx':['-g','-w',"-O0","-include","st_spix/csrc/prop/pch.h"],
        #                     'nvcc':['-w','-G',"-include", "st_spix/csrc/prop/pch.h"]},
        #extra_link_args=['-Wl,-Bstatic', '-lpthread']),  # Linker flags if needed

    ],
    # cmdclass={'build_ext': build_ext},
    cmdclass={'build_ext': BuildExtension},
)
