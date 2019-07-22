import sys
import os
sys.path.append('/home/s1680171/wave_propagation/')
sys.path.append('/mnt/mscteach_home/s1680171/wave_propagation/')
sys.path.append(os.path.dirname(os.getcwd()))


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


experiments_to_test = [
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001",c
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_16h",
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_16h_2",
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_16h_c",
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_correct_code",
                        # "ar_lstm_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_dataset_fixed_tub",
                        "convlstm_batch_8_samples_5_in_5_out_10_normal_lr_0.001_dataset_fixed_tub",
                        "predrnn_batch_4_samples_1_in_5_out_10_normal_lr_0.0001",
                        "predrnn_batch_4_samples_1_in_5_out_20_normal_lr_0.0001",
                        "predrnn_batch_4_samples_5_in_5_out_5_normal_lr_0.0001",
                        "predrnn_batch_4_samples_5_in_5_out_10_normal_lr_0.0001",
]

partition = 'Short'
time = '0-03:59:59'

with open('test.template', 'r') as file:
    template = file.read()
file.close()

exp_args = {"test_starting_point": 15,
            "num_total_output_frames": 80,
            "get_sample_predictions": "True",
            "belated": "True"
            }

for exp_name in experiments_to_test:
    args_template = "--experiment_name {exp_name} --test_starting_point {test_starting_point} --num_total_output_frames {num_total_output_frames} --get_sample_predictions {get_sample_predictions} --belated {belated}"

    args_template = args_template.format_map(SafeDict(exp_name=exp_name))
    args_template = args_template.format(**exp_args)
    exp_template = template.format_map(SafeDict(args=args_template, partition=partition, time=time))
    exp_script = 'test_%s_sp_%d.sh' % (exp_name, exp_args['test_starting_point'])
    with open(exp_script, 'w') as file:
        file.write(exp_template)
    file.close()

    print('Running test for %s' % exp_name)
    os.system("sbatch -o test_%s_%%j.out %s" % (exp_script, exp_script))