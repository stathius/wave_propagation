#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Short
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=03:59:59
#SBATCH --exclude=marax # 980 with 4gb
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
	rsync -ua --progress /home/s1680171/wave_propagation/video.tar.gz /disk/scratch/s1680171/wave_propagation/
	echo 'unzipping'
	tar zxfk /disk/scratch/s1680171/wave_propagation/video.tar.gz -C /disk/scratch/s1680171/wave_propagation/ >/dev/null 2>&1
	echo 'unzipping finished'
else
	echo 'dataset already uploaded'
fi

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd /home/s1680171/wave_propagation/
python test_network.py --experiment_name AR_LSTM_batch16_samples10_epoch_25_out20
python test_network.py --experiment_name AR_LSTM_batch16_samples10_epoch_25_out20_2

python test_network.py --experiment_name AR_LSTM_AR_LSTM_batch16_samples10_epoch_25_out20_unnorm
python test_network.py --experiment_name AR_LSTM_AR_LSTM_batch16_samples10_epoch_25_out20_unnorm_2
