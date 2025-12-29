
### Setup

```
cd <to this dir>
conda create --name cuda_try python=3.11
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# To make sure to get the lib10.so in the LD_LIBRARY_PATH to load compiled .so file
cat > $CONDA_PREFIX/etc/conda/activate.d/torch_libs.sh <<'SH'
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH}"
SH
conda activate cuda_try
```

### Compile the extention
```
pip install ninja # for faster build

python setup.py install
```

### Make sure the extention library loads properly
```
python -c "import cust_add_ext; print(cust_add_ext.__file__)"
/home/ubuntu/miniconda/envs/cuda_try/lib/python3.11/site-packages/cust_add_ext.cpython-311-x86_64-linux-gnu.so
```

### Now run the [test.ipynb](test.ipynb) 

The first time it will take sometime to load the 
1. CUDA context creation (biggest usual one)
2. The first time your program touches CUDA, the driver has to: initialize the CUDA runtime and create a CUDA context for the process 
3. set up allocator state, streams, cuBLAS/cuDNN handles (sometimes lazily)

2) Dynamic linking + loading your extension .so

On the first import/use of cust_add_ext:
1. the shared library is loaded (dlopen)
2. all dependent shared libs load too (PyTorch CUDA libs are large)
3. symbols are resolved

3) PTX JIT compilation (if you shipped PTX / wrong arch)
If your extension contains PTX that needs to be compiled to SASS for the GPU, the driver will JIT it on first use.
This is more likely if:
1. you compiled only compute_XX (PTX) and not sm_XX or you compiled for a different GPU arch than the current machine

So you should have native SASS, but the presence of PTX can still trigger some JIT paths in certain situations. (Less likely though.)

4) PyTorch CUDA caching allocator warm-up

First allocation on CUDA can be slow; later allocations are fast because memory is cached.

5) Async timing illusion (very common)

CUDA ops are asynchronous. If you “time” the first call naively, you might accidentally include:

the import time

build/load time

context creation

implicit synchronizations from later ops

To measure kernel runtime you must synchronize.



