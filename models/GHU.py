import torch
import torch.nn as nn


class GHU(nn.Module):
    def __init__(self, filter_size, num_features, input_channels, device):
        """Initialize the Gradient Highway Unit.
        """
        super(GHU, self).__init__()
        self.filter_size = filter_size
        self.num_features = num_features
        self.input_channels = input_channels
        self.conv_z = nn.Conv2d(in_channels=self.num_features,
                                out_channels=self.num_features * 2,
                                kernel_size=self.filter_size,
                                stride=1,
                                padding=1)
        self.conv_x = nn.Conv2d(in_channels=self.input_channels,
                                out_channels=self.num_features * 2,
                                kernel_size=self.filter_size,
                                stride=1,
                                padding=1)
        self.device = device

    def init_state(self, inputs, num_features):
        dims = len(inputs.shape)
        if dims == 4:
            batch = inputs.shape[0]
            height = inputs.shape[2]
            width = inputs.shape[3]
        else:
            raise ValueError('input tensor should be rank 4.')
        return torch.zeros([batch, num_features, height, width]).to(self.device)

    def forward(self, x, z):
        if z is None:
            z = self.init_state(x, self.num_features)
        z_concat = self.conv_z(z)

        x_concat = self.conv_x(x)
        # if self.layer_norm:
        #     x_concat = tensor_layer_norm(x_concat, 'input_to_state')

        gates = x_concat + z_concat
        p, u = torch.chunk(gates, 2, dim=1)
        p = torch.tanh(p)
        u = torch.sigmoid(u)
        z_new = u * p + (1 - u) * z
        return z_new
