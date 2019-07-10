import sys
import os
sys.path.append('/home/s1680171/wave_propagation/')
sys.path.append('/mnt/mscteach_home/s1680171/wave_propagation/')
sys.path.append(os.path.dirname(os.getcwd()))
from utils.io import load_json


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


experiments_to_test = [
    "AR_LSTM_batch_16_samples_10_in_5_out_20_none_lr_0.0001",
    "AR_LSTM_batch_16_samples_10_in_5_out_20_none_lr_0.0001_2",
    "AR_LSTM_batch_16_samples_10_in_5_out_20_normal_lr_0.0001",
    "AR_LSTM_batch_16_samples_10_in_5_out_20_normal_lr_0.0001_2",
    "ConvLSTM_batch_16_samples_5_epoch30_in_5_out_5_lr_0.001",
    "ConvLSTM_batch_16_samples_5_in_5_out_1_lr_0.001",
    "ConvLSTM_batch_16_samples_5_in_5_out_5_lr_0.001",
    "ConvLSTM_batch_6_samples_10_in_5_out_20_normal_lr_0.001",
    "ConvLSTM_batch_6_samples_5_in_5_out_20_normal_lr_0.001",
    "ConvLSTM_batch_8_samples_5_out_10_normal_lr_0.001",
    "ResNet_batch_16_samples_5_in_5_out_10_lr_0.001",
    "ResNet_batch_16_samples_5_in_5_out_1_lr_0.001",
    "ResNet_batch_16_samples_5_in_5_out_1_lr_0.001_2",
    "ResNet_batch_16_samples_5_in_5_out_5_lr_0.001",
    "ar_lstm_batch_16_samples_10_epoch_25_in_5_out_20_normalizer_m1to1_lr_0.0001",
    "ar_lstm_batch_16_samples_10_epoch_25_in_5_out_20_normalizer_none_lr_0.0001",
    "ar_lstm_batch_16_samples_10_epoch_25_in_5_out_20_normalizer_normal_lr_0.0001",
    "convlstm_batch_6_samples_3_in_5_out_20_normal_lr_0.001",
    "convlstm_batch_6_samples_5_in_2_out_20_normal_lr_0.001",
    "resnet_batch_16_samples_10_in_5_out_30_lr_0.001",
    "resnet_batch_16_samples_5_in_10_out_20_lr_0.001",
    "resnet_batch_16_samples_in_3_out_10_lr_0.001"
]

with open('script_template_test.sh', 'r') as file:
    template = file.read()
file.close()

exp_args = {"test_starting_point": 15,
            "num_total_output_frames": 80}

for exp_name in experiments_to_test:
    # # for arg, arg_val in :

    args_template = "--experiment_name {exp_name} --test_starting_point {test_starting_point} --num_total_output_frames {num_total_output_frames}"

    args_template = args_template.format_map(SafeDict(exp_name=exp_name))
    args_template = args_template.format(**exp_args)
    exp_template = template.format_map(SafeDict(args=args_template))
    exp_script = 'test_%s.sh' % exp_name
    with open(exp_script, 'w') as file:
        file.write(exp_template)
    file.close()

    os.system("sbatch -o test_%s_%%j.out %s" % (exp_name, exp_script))