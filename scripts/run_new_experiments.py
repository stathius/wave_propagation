import sys
import os
sys.path.append('/home/s1680171/wave_propagation/')
sys.path.append('/mnt/mscteach_home/s1680171/wave_propagation/')
sys.path.append(os.path.dirname(os.getcwd()))
from utils.io import load_json


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


new_experiments = load_json('new_experiments.json')

with open('script_template.sh', 'r') as file:
    template = file.read()
file.close()

for exp, exp_args in new_experiments.items():
    print(exp, exp_args)
    # for arg, arg_val in exp_args:
    exp_name = '{model_type}_batch_{batch_size}_samples_{samples_per_sequence}_epoch_{num_epochs}_in_{num_input_frames}_out_{num_output_frames}_normalizer_{normalizer_type}_lr_{learning_rate}'.format(**exp_args)

    args_template = "--experiment_name {exp_name} --model_type {model_type} --batch_size {batch_size} --num_epochs {num_epochs} --samples_per_sequence {samples_per_sequence} --num_input_frames {num_input_frames} --num_output_frames {num_output_frames} --learning_rate {learning_rate} --normalizer_type {normalizer_type} --num_workers 12"

    args_template = args_template.format_map(SafeDict(exp_name=exp_name))
    args_template = args_template.format(**exp_args)
    exp_template = template.format_map(SafeDict(args=args_template))
    exp_script = '%s.sh' % exp_name
    with open(exp_script, 'w') as file:
        file.write(exp_template)
    file.close()

    # os.system("sbatch -o %s_%%j.out %s" % (exp_name, exp_script))