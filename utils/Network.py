import torch.nn as nn
import torch

class Network (nn.Module):
    """
    The network structure
    """
    def __init__(self, device, channels):
        super(Network, self).__init__()
        self.channels = channels
        self.device = device
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5*channels, 60, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Conv2d(60, 120, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.Conv2d(120, 240, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.Conv2d(240, 480, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=480),
            nn.Tanh(),
            nn.Dropout2d(0.25)
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(30720, 1000),
            nn.Tanh(),
            nn.Dropout(0.25)
        )

        self.decoder_linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1000, 30720)
        )

        self.decoder_conv = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose2d(480, 240, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=240),
            nn.Tanh(),
            nn.ConvTranspose2d(240, 120, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=120),
            nn.Tanh(),
            nn.ConvTranspose2d(120, 60, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(num_features=60),
            nn.Tanh(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(60, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_0 = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)

        self.LSTM = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)

        self.LSTM_new_input = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)


    def forward(self, x, mode="input", training=False): #"input", "new_input", "internal"
        x.requires_grad_(training)
        with torch.set_grad_enabled(training):
            if "input" in mode:
                x = self.encoder_conv(x)
                self.org_size = x.size()
                x = x.view(-1, 30720)
                x = self.encoder_linear(x)
                if mode == "input":
                    self.h0, self.c0 = self.LSTM_0(x, (self.h0, self.c0))
                elif mode == "new_input":
                    self.h0, self.c0 = self.LSTM_new_input(x, (self.h0, self.c0))
            elif mode == "internal":
                self.h0, self.c0 = self.LSTM(self.h0, (self.h0, self.c0))
            x = self.h0.clone()
            x = self.decoder_linear(x)
            x = x.view(self.org_size)
            x = self.decoder_conv(x)
            return x

    def reset_hidden(self, batch_size, training=False):
        self.h0 = torch.zeros((batch_size, 1000), requires_grad=training).to(self.device) #Requires grad replaces Variable
        self.c0 = torch.zeros((batch_size, 1000), requires_grad=training).to(self.device)