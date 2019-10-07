import sys
import os
sys.path.append('/home/s1680171/wave_propagation/')
sys.path.append('/mnt/mscteach_home/s1680171/wave_propagation/')
sys.path.append(os.path.dirname(os.getcwd()))
from utils.io import load_json


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


# These are with fixed tub, rerun them with original data
# train_ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001.sh
#   train_convlstm_batch_8_samples_5_in_5_out_10_normal_lr_0.001.sh
#    train_resnet_batch_16_samples_5_in_5_out_10_normal_lr_0.001.sh

new_experiments = load_json('new_experiments.json')

with open('train.template', 'r') as file:
    template = file.read()
file.close()

for exp, exp_args in new_experiments.items():
    print(exp, exp_args)
    # for arg, arg_val in exp_args:
    exp_name = '{model_type}_batch_{batch_size}_samples_{samples_per_sequence}_in_{num_input_frames}_out_{num_output_frames}_{normalizer_type}_lr_{learning_rate}_dataset_{dataset}_{time_readable}_patience_{scheduler_patience}_back_and_forth_{back_and_forth}'.format(**exp_args)

    args_template = "--experiment_name {exp_name} --model_type {model_type} --batch_size {batch_size} --num_epochs {num_epochs} --samples_per_sequence {samples_per_sequence} --num_input_frames {num_input_frames} --num_output_frames {num_output_frames} --learning_rate {learning_rate} --normalizer_type {normalizer_type} --dataset {dataset} --scheduler_patience {scheduler_patience} --num_workers 12 --back_and_forth {back_and_forth}"

    args_template = args_template.format_map(SafeDict(exp_name=exp_name))
    args_template = args_template.format(**exp_args)
    exp_template = template.format_map(SafeDict(exp_args))
    exp_template = exp_template.format_map(SafeDict(args=args_template))
    exp_script = 'train_%s.sh' % exp_name
    with open(exp_script, 'w') as file:
        file.write(exp_template)
    file.close()

    os.system("sbatch -o %s_%%j.out %s" % (exp_script, exp_script))