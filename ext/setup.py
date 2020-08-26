from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='zyan',
    ext_modules=[
        CppExtension('zyan', ['person_face_cost.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
