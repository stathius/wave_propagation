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

exp_name = '{model_type}_batch_{batch_size}_samples_{samples_per_sequence}_epoch_{num_epochs}_in_{num_input_frames}_out_{num_output_frames}'

for exp, exp_args in new_experiments.items():
    # for arg, arg_val in exp_args:
    exp_name = '{model_type}_batch_{batch_size}_samples_{samples_per_sequence}_epoch_{num_epochs}_in_{num_input_frames}_out_{num_output_frames}'.format(**exp_args)
    template = template.format_map(SafeDict(exp_args))
    template = template.format_map(SafeDict(exp_name=exp_name))
    exp_script = '%s.sh' % exp_name
    with open(exp_script, 'w') as file:
        template = file.write(template)
    file.close()

    os.system("sbatch -o %s_%%j.out %s" % (exp_name, exp_script))