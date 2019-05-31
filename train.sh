#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=MSC
#SBATCH --gres=gpu:2
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-24:00:00
hostname
nvidia-smi
export CUDA_HOME=/opt/cuda-9.0.176.1/
export CUDNN_HOME=/opt/cuDNN-7.0/
export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDNN_HOME}/include:$CPATH
export PATH=${CUDA_HOME}/bin:${PATH}
export PYTHON_PATH=$PATH

export STUDENT_ID=$(whoami)
mkdir -p /disk/scratch/s1680171/wave_propagation

export TMPDIR=/disk/scratch/s1680171/

file=/disk/scratch/s1680171/wave_propagation/video.tar.gz
if [ ! -f $file ]
then
	echo 'sending tar file'
	rsync -ua --progress /home/s1680171/wave_propagation/video.tar.gz /disk/scratch/s1680171/wave_propagation
	echo 'unzipping'
	tar zxfk /disk/scratch/s1680171/wave_propagation/video.tar.gz -C /disk/scratch/s1680171/wave_propagation/ >/dev/null 2>&1
	echo 'unzipping finished'
else
	echo 'dataset already uploaded'
fi

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
python train.py