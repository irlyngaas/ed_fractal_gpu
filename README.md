# Fast renderer for FractalDB using GPUs.

**Author:** Edgar MARTINEZ

FractalDB code based on the original idea: [Pretraining without Natural Images](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)

## Requirements on ABCI

This repository has been tested on GPUs using Volta nodes.
- **Modules:**
    1) cuda/12.2/12.2.0
    2) cudnn/8.9/8.9.2
    3) nccl/2.18/2.18.5-1
    4) gcc/12.2.0
    5) cmake/3.26.1
    6) hpcx-mt/2.12

**Python version:**
    Python 3.11.1 (main, Feb 2 2023) [GCC 12.2.0] on Linux.

We chose Python 3.11.1 due to a 20% improvement in certain CPU operations compared to version 3.10. We use "pyenv" to manage multiple environments, which can be found at [pyenv GitHub](https://github.com/pyenv/pyenv).

We recommend installing the packages in the following order:

1) Install Python 3.11.1
    1.1 Update pip

2) Install the remaining packages from the "requirements.txt" file.
    pip install -r requirements.txt

3) Untar the CSV files.
    tar -xvf csv/data1k_fromPython.tar -C csv/

## Usage

We used an interactive V100 with the following command: 
```
qrsh -g gcc50533 -l rt_F=1 -l h_rt=02:00:00 -l USE_SSH=1 -v SSH_PORT=2299
```

1) Clone this repository or copy it to your desired location.
2) Locate your local SSD and create a symbolic link to your workspace:
   ```
   ln -s $SGE_LOCALDIR ssd
   ```

### Creating a Large FractalDB Using GPUs (Headless Rendering Using EGL)

Use the following command to create FractalDB on the local SSD. You can modify the "save_root" option to save the dataset in a different location.
```
mpirun --bind-to socket --use-hwthread-cpus -np 80 python mpi_createFDB_gpu.py -g 4 --image_res 362 --iteration 100000 --save_root ssd/fdb1k
```

This will create FractalDB-1k, including 1 million images. If you want to create more instances, modify the --instance flag.

To use multi-node on ABCI:
```
qsub -g xxxxxx -l USE_SSH=1 -v SSH_PORT=2299 multi_node.sh
```

Note: You can use the normal render way to a window on a local machine(NOT HEADLESS) using GLFW. Just modify the --backend option.
```
mpirun --bind-to socket --use-hwthread-cpus -np 10 python mpi_createFDB_gpu.py -g 1 --image_res 362 --iteration 100000 --save_root ssd/fdb1k --backend glfw
```