import torch.nn as nn
import torch



class MyDataParallel(nn.DataParallel):
    # def __init__(self, model):
        # super(MyDataParallel, self).__init__()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    # def __getattr__(self, name):
        # return getattr(self.module, name)

class Network(nn.Module):
    """
    The network structure
    """
    def __init__(self, num_channels):
        super(Network, self).__init__()
        self.num_channels = num_channels
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(5*num_channels, 60, kernel_size=7, stride=2, padding=1),
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
            nn.ConvTranspose2d(60, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.LSTM_initial_input = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)
        self.LSTM_propagation = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)
        self.LSTM_reinsert = nn.LSTMCell(input_size=1000, hidden_size=1000, bias=True)


    def forward(self, x, mode="initial_input", training=False): #"initial_input", "new_initial_input", "internal"
        x.requires_grad_(training)
        with torch.set_grad_enabled(training):
            if "initial_input" in mode:
                x = self.encoder_conv(x)
                self.org_size = x.size()
                x = x.view(-1, 30720)
                x = self.encoder_linear(x)
                if mode == "initial_input":
                    self.h0, self.c0 = self.LSTM_initial_input(x, (self.h0, self.c0))
                elif mode == "reinsert":
                    self.h0, self.c0 = self.LSTM_reinserting(x, (self.h0, self.c0))
            elif mode == "propagate":
                self.h0, self.c0 = self.LSTM_propagation(self.h0, (self.h0, self.c0))
            x = self.h0.clone()
            x = self.decoder_linear(x)
            x = x.view(self.org_size)
            x = self.decoder_conv(x)
            return x

    def reset_hidden(self, batch_size, training=False):
        # TODO
        # user random values?
        self.h0 = torch.zeros((batch_size, 1000), requires_grad=training) #Requires grad replaces Variable
        self.c0 = torch.zeros((batch_size, 1000), requires_grad=training)