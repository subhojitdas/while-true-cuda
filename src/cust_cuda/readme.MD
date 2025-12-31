
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

----
### See the PTX from the .cu file directly

Run the below , it will generate the ptx file.
```
export TORCH_INC=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include"))')
export TORCH_API_INC=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "include", "torch", "csrc", "api", "include"))')
export PY_INC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

nvcc -ptx cust_add_cuda.cu -o cust_add_cuda.ptx \
  -I"$TORCH_INC" -I"$TORCH_API_INC" -I"$PY_INC" \
  -I/usr/local/cuda/include \
  -std=c++17
```

### See the PTX and SASS code from the .so generated after compilation

Compiled .so file can be found in `/home/ubuntu/miniconda/envs/cuda_try/lib/python3.11/site-packages/cust_add_ext-0.0.0-py3.11-linux-x86_64.egg/cust_add_ext.cpython-311-x86_64-linux-gnu.so`

```
ELF .so
 ├── host C++ code
 ├── CUDA fatbinary
 │    ├── SASS (sm_86)
 │    └── PTX (compute_86)
 └── symbols
```

If `.so` has the SASS code it will show something like this -
```
Fatbin elf code:
================
arch = sm_86 # Native SASS exists
arch = compute_86 # PTX fallback
```

use `cuobjdump` ,
```
cuobjdump --dump-ptx cust_add_ext.cpython-311-x86_64-linux-gnu.so
cuobjdump --dump-sass cust_add_ext.cpython-311-x86_64-linux-gnu.so > sass_lineinfo.txt
```

```
head -n 35 sass.txt

Fatbin elf code:
================
arch = sm_86
code version = [1,7]
host = linux
compile_size = 64bit

	code for sm_86
		Function : _Z10add_kernelPKfPffi
	.headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM86 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM86)"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                        /* 0x00000a0000017a02 */
                                                                                 /* 0x000fe40000000f00 */
        /*0010*/                   S2R R4, SR_CTAID.X ;                          /* 0x0000000000047919 */
                                                                                 /* 0x000e280000002500 */
        /*0020*/                   S2R R3, SR_TID.X ;                            /* 0x0000000000037919 */
                                                                                 /* 0x000e240000002100 */
        /*0030*/                   IMAD R4, R4, c[0x0][0x0], R3 ;                /* 0x0000000004047a24 */
                                                                                 /* 0x001fca00078e0203 */
        /*0040*/                   ISETP.GE.AND P0, PT, R4, c[0x0][0x174], PT ;  /* 0x00005d0004007a0c */
                                                                                 /* 0x000fda0003f06270 */
        /*0050*/               @P0 EXIT ;                                        /* 0x000000000000094d */
                                                                                 /* 0x000fea0003800000 */
        /*0060*/                   MOV R5, 0x4 ;                                 /* 0x0000000400057802 */
                                                                                 /* 0x000fe20000000f00 */
        /*0070*/                   ULDC.64 UR4, c[0x0][0x118] ;                  /* 0x0000460000047ab9 */
                                                                                 /* 0x000fc80000000a00 */
        /*0080*/                   IMAD.WIDE R2, R4, R5, c[0x0][0x160] ;         /* 0x0000580004027625 */
                                                                                 /* 0x000fcc00078e0205 */
        /*0090*/                   LDG.E R2, [R2.64] ;                           /* 0x0000000402027981 */
                                                                                 /* 0x000ea2000c1e1900 */
        /*00a0*/                   IMAD.WIDE R4, R4, R5, c[0x0][0x168] ;         /* 0x00005a0004047625 */
                                                                                 /* 0x000fe200078e0205 */
        /*00b0*/                   FADD R7, R2, c[0x0][0x170] ;                  /* 0x00005c0002077621 */
                                                                                 /* 0x004fca0000000000 */
```

### Reading symbols table from the .so file symbols file:

```
> nm -D cust_add_ext.cpython-311-x86_64-linux-gnu.so > symbols.txt

                 U PyGILState_GetThisThreadState
                 U PyGILState_Release
000000000004a990 T PyInit_cust_add_ext
                 U PyInstanceMethod_New
```

- `T` meaning the method is defined in this .so
- `U` meaning the symbols need to be loaded from a different `.so` file
- 

#### Symbol Table:

```
> readelf -s cust_add_ext.cpython-311-x86_64-linux-gnu.so

  6299: 00000000000871ce   110 FUNC    WEAK   DEFAULT   16 _ZSt13__copy_mov[...]
  6300: 00000000000666b6    31 FUNC    WEAK   DEFAULT   16 _ZNSt8__detail15[...]
  6301: 00000000000849fc    55 FUNC    WEAK   DEFAULT   16 _ZNSt16_Sp_count[...]
  6302: 000000000006cc76    98 FUNC    WEAK   DEFAULT   16 _ZNSt10unique_pt[...]
  6303: 000000000006a8fa    90 FUNC    WEAK   DEFAULT   16 _ZNSt6vectorISt8[...]
  6304: 0000000000000000     0 OBJECT  GLOBAL DEFAULT  UND _ZTISt11range_er[...]
```

