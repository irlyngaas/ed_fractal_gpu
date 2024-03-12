#!/bin/bash
#$ -cwd
#$ -l rt_F=4
#$ -l h_rt=01:00:00
#$ -j y
#$ -o output/$JOB_ID_fdb1k_test_rendertossd.out
#$ -N render_test_fdb1k
#$ -l USE_BEEOND=1
cat $JOB_SCRIPT
echo "..................................................................................................."
echo "JOB ID: ---- >>>>>>   $JOB_ID"
# ======== Modules ========
source /etc/profile.d/modules.sh
module purge
module load cuda/12.2/12.2.0 cudnn/8.9/8.9.2 nccl/2.18/2.18.5-1 gcc/12.2.0 cmake/3.26.1 hpcx-mt/2.12

# ======== Pyenv/ ========
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
pyenv local tested


export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

############# Render to local SSD
# export SSD=/local/${JOB_ID}.1.gpu
# export LOCALDIR=${SSD}
export LOCALDIR=/beeond
export RENDER_HWD=egl
export DATASET=${LOCALDIR}/fdb1k_${RENDER_HWD}

cd render_engines/fdb
echo "Start untar to local ..."

time tar -xf csv/data1k_fromPython.tar -C ${LOCALDIR}

echo "Start REDNERING to local ..."
# For GPU
mpirun --bind-to socket -machinefile $SGE_JOB_HOSTLIST --use-hwthread-cpus -npernode 80 -np 320 python mpi_createFDB_gpu.py --image_res 362 --ngpus-pernode 4 --save_root /beeond/fdb1k --load_root /beeond/data1k_fromPython/csv_rate0.2_category1000

##### Debug local
echo "Content of the local SSD"
readlink -f ${LOCALDIR}
ls ${LOCALDIR}

echo "Classes on FractalDB"
ls ${LOCALDIR}/fdb1k_egl 

echo "Total number of images:"
find ${LOCALDIR}/fdb1k_egl -type f -print |wc -l