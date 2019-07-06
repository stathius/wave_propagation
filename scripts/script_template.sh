#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time={time}
#SBATCH --exclude=marax # 980 with 4gb
hostname
nvidia-smi
export LD_LIBRARY_PATH=/opt/cuDNN-7.0/lib64:/opt/cuda-9.0.176.1//lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/cuDNN-7.0/lib64:$LIBRARY_PATH
export CPATH=/opt/cuDNN-7.0/include:$CPATH
export PATH=/opt/cuda-9.0.176.1//bin:${PATH}
export PYTHON_PATH=$PATH

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

source /home/s1680171/miniconda3/bin/activate mlp
cd /home/s1680171/wave_propagation/

python train_network.py --experiment_name {exp_name} --model_type {model_type} --batch_size {batch_size} --num_epochs {num_epochs} --samples_per_sequence {samples_per_sequence} --num_input_frames {num_input_frames} --num_output_frames {num_output_frames} --num_workers 12