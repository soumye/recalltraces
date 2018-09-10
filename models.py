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

# The State Generator. P(Δs_t | a_t, s_t+1).
class StateGen(nn.Module):
    def __init__(self, num_actions):
        super(StateGen, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.hidden_units = 32 * 7 * 7
        self.fc4 = nn.Linear(self.hidden_units, 512)

        self.fc_encode = nn.Linear(512, 512)
        self.fc_action = nn.Linear(num_actions, 512)
        self.fc_decode = nn.Linear(512, 512)

        self.fc5 = nn.Linear(512, self.hidden_units)
        self.deconv6 = nn.ConvTranspose2d(32, 64, 3, stride=1)
        self.deconv7 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv8 = nn.ConvTranspose2d(32, 4, 8, stride=4)

        self.init_weights()
        # # start to do the init...
        # nn.init.orthogonal_(self.conv1.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.conv2.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.conv3.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.fc4.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.fc_encode.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.fc_action.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.fc_decode.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.fc5.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.deconv6.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.deconv7.weight.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.deconv8.weight.data, gain=nn.init.calculate_gain('relu'))

        # # init the bias...ob
        # nn.init.constant_(self.conv1.bias.data, 0)
        # nn.init.constant_(self.conv2.bias.data, 0)
        # nn.init.constant_(self.conv3.bias.data, 0)
        # nn.init.constant_(self.fc4.bias.data, 0)
        # nn.init.constant_(self.fc_encode.bias.data, 0)
        # nn.init.constant_(self.fc_action.bias.data, 0)
        # nn.init.constant_(self.fc_decode.bias.data, 0)
        # nn.init.constant_(self.fc5.bias.data, 0)
        # nn.init.constant_(self.deconv6.bias.data, 0)
        # nn.init.constant_(self.deconv7.bias.data, 0)
        # nn.init.constant_(self.deconv8.bias.data, 0)
    
    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight.data)
            nn.init.constant_(layer.bias.data, 0)
        nn.init.uniform_(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform_(self.fc_action.weight.data, -0.1, 0.1)
        
    def forward(self, obs, action):
        """
        obs : #batchsize x 4 x 84x84
        action : #batchsize x #num_action(one_hot)
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.hidden_units)
        x = F.relu(self.fc4(x))
        x = self.fc_encode(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc_decode(x)
        x = F.relu(self.fc5(x))
        x = x.view((-1, 32, 7, 7))
        x = F.relu(self.deconv6(x))
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x
