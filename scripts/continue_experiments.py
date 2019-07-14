import sys
import os
sys.path.append('/home/s1680171/wave_propagation/')
sys.path.append('/mnt/mscteach_home/s1680171/wave_propagation/')
sys.path.append(os.path.dirname(os.getcwd()))


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


with open('train.template', 'r') as file:
    template = file.read()
file.close()

experiment_names = ['ar_lstm_batch_16_samples_10_epoch_25_in_5_out_20_normalizer_m1to1_lr_0.0001']


num_epochs = 100
sbatch_args = {"partition": "Standard",
               "time": "0-07:59:59"}


for exp_name in experiment_names:
    print(exp_name)

    args_template = "--experiment_name {exp_name} --num_epochs %d --continue_experiment True" % num_epochs

    args_template = args_template.format_map(SafeDict(exp_name=exp_name))
    exp_template = template.format_map(SafeDict(sbatch_args))
    exp_template = exp_template.format_map(SafeDict(args=args_template))
    exp_script = '%s.sh' % exp_name
    with open(exp_script, 'w') as file:
        file.write(exp_template)
    file.close()

    os.system("sbatch -o %s_%%j.out %s" % (exp_name, exp_script))