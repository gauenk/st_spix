
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
        CUDAExtension('st_spix_cuda', [
            'st_spix/pwd/pair_wise_distance_cuda_source.cu',
        ],extra_compile_args={'cxx': ['-g','-w'],'nvcc': ['-O2','-w']})
    ],
    cmdclass={'build_ext': BuildExtension},
)
