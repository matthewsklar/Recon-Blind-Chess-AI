import torch
import torch.nn as nn
import torch.nn.functional as F
import msklar3_mdiamond8_chess_helper as helper
import msklar3_mdiamond8_config as config


class Net(nn.Module):
    def __init__(self, in_channels, policy_size):
        super(Net, self).__init__()
        
        self.in_channels = in_channels

        self.ResidualLayer()
        self.PolicyNet(policy_size)
        self.ValueNet()

    def forward(self, x, legal_moves):
        # x = self.ResidualForward(x)
        policy = self.PolicyForward(x, legal_moves)
        value = self.ValueForward(x)

        return policy, value

    def ResidualLayer(self):
        # Convolution Network: 256 filters, kernel size 3x3, stride 1
        self.residual_conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Batch normalizer
        self.residual_batchnorm1 = nn.BatchNorm2d(256)

        # Convolution Network: 256 filters, kernel size 3x3, stride 1
        self.residual_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            stride=1,
            bias=False
        )

        # Batch normalizer
        self.residual_batchnorm2 = nn.BatchNorm2d(256)

    def PolicyNet(self, policy_size):
        # Convolutional Network: 2 filters, kernel size 1x1, stride 1
        self.policy_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=2,
            kernel_size=1,
            stride=1,
            bias=False)

        # Batch normalizer
        self.policy_batchnorm = nn.BatchNorm2d(2)

        # Fully connected linear layer
        self.policy_linear = nn.Linear(192, policy_size, False)

        # Softmax
        self.policy_softmax = nn.Softmax(0)

    def ValueNet(self):
        # Convolutional Network: 1 filter, kernel size 1x1, stride 1
        self.value_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False)

        # Batch normalizer
        self.value_batchnorm = nn.BatchNorm2d(1)

        # Linear Hidden layer
        self.value_linear1 = nn.Linear(96, 1, False)

        # Linear output value layer
        self.value_linear2 = nn.Linear(1, 1, False)

    def ResidualForward(self, x):
        og_x = x

        # Convolutional layer 1
        x = self.residual_conv1(x.float())
        x = F.leaky_relu(self.residual_batchnorm1(x))

        # Convolutional layer 2
        x = self.residual_conv2(x.float())
        x = self.residual_batchnorm2(x)
        print('x', x)
        print('og_x', og_x)
        x = x + og_x
        x = nn.LeakyReLU(x)

        return x

    def PolicyForward(self, x, legal_moves):
        # Convolutional layer
        x = self.policy_conv(x.float())
        x = F.leaky_relu(self.policy_batchnorm(x))

        # Linear layer
        x = self.policy_linear(x.flatten())
        x = self.policy_softmax(x)
        x = x * torch.tensor(legal_moves)    # Filter out illegal moves

        return x

    def ValueForward(self, x):
        # Convolutional layer
        x = self.value_conv(x.float())
        x = F.leaky_relu(self.value_batchnorm(x))

        # Linear layer
        x = F.leaky_relu(self.value_linear1(x.flatten()))

        # Linear layer
        x = torch.tanh(self.value_linear2(x.flatten()))

        return x

    def loss(self, v, v_hat, pi, p):
        for i, probability in enumerate(p):
            if probability <= 1 - config.LOG_EPSILON:
                p[i] += config.LOG_EPSILON

        prediction_mse = torch.sum((v - v_hat)**2)
        probability_log = torch.sum(pi * torch.log(p))
        l2_norm = config.LAMBDA * (torch.norm(p) ** 2)

        return prediction_mse - probability_log + l2_norm
