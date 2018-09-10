import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

# the convolution layer of deepmind
class DeepMind(nn.Module):
    def __init__(self):
        super(DeepMind, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.hidden_units = 32 * 7 * 7
        self.fc1 = nn.Linear(self.hidden_units, 512)
        
        # start to do the init...
        nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc1.weight.data, gain=nn.init.calculate_gain('relu'))
        # init the bias...
        nn.init.constant_(self.conv1.bias.data, 0)
        nn.init.constant_(self.conv2.bias.data, 0)
        nn.init.constant_(self.conv3.bias.data, 0)
        nn.init.constant_(self.fc1.bias.data, 0)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.hidden_units)
        x = F.relu(self.fc1(x))

        return x

# in the initial, just the nature CNN
class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.cnn_layer = DeepMind()
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        # init the linear layer..
        nn.init.orthogonal_(self.critic.weight.data)
        nn.init.constant_(self.critic.bias.data, 0)
        # init the policy layer...
        nn.init.orthogonal_(self.actor.weight.data, gain=0.01)
        nn.init.constant_(self.actor.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        value = self.critic(x)
        pi = F.softmax(self.actor(x), dim=1)

        return value, pi

# The action generator. P(a_t | s_t+1)
class ActGen(nn.Module):
    def __init__(self, num_actions):
        super(ActGen, self).__init__()
        self.cnn_layer = DeepMind()
        self.actgen = nn.Linear(512, num_actions)
        # init the policy layer...
        nn.init.orthogonal_(self.actgen.weight.data, gain=0.01)
        nn.init.constant_(self.actgen.bias.data, 0)

    def forward(self, inputs):
        x = self.cnn_layer(inputs / 255.0)
        pi = F.softmax(self.actgen(x), dim=1)
        return pi