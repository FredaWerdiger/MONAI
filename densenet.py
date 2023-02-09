import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class conv_block(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 dropout_rate,
                 bottleneck=False):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_rate),
        )
        self.bottleneck = bottleneck
        self.bottleneck_block = nn.Sequential(
            nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in, ch_out * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(ch_out * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out * 4, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        if self.bottleneck:
            x = self.bottleneck_block(x)
        else:
            x = self.conv(x)
        return x


class dense_block(nn.Module):
    def __init__(self,
                 ch_in,
                 nb_layers,
                 growth_rate,
                 dropout_rate=0.2,
                 bottleneck=False,
                 grow_nb_filters=True,
                 return_concat_list=False
                 ):
        super(dense_block, self).__init__()
        self.nb_layers = nb_layers
        self.growth_rate = growth_rate
        self.nb_filters = ch_in
        self.dropout = dropout_rate
        self.bottleneck = bottleneck
        self.return_concat_list = return_concat_list
        self.grow_nb_filters = grow_nb_filters
        self.cbs = []
        for layer in range(nb_layers):
            filter_in = ch_in + (layer * growth_rate)
            self.cb = conv_block(filter_in, self.growth_rate, self.dropout, self.bottleneck)
            self.cbs.append(self.cb)

    def forward(self, x):
        x_list = [x]
        for conv in self.cbs:
            cb = conv(x)
            x_list.append(cb)
            x = torch.concat((x, cb), dim=1)
            # print(x.shape)
            if self.grow_nb_filters:
                self.nb_filters += self.growth_rate  # number of filters grows by growth rate in each layer
        if self.return_concat_list:
            return x, x_list
        else:
            return x


class transition_down(nn.Module):
    def __init__(self, ch_in):
        super(transition_down, self).__init__()
        self.down_block = nn.Sequential(
            nn.BatchNorm3d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_in, ch_in, (1, 1, 1), padding=0, stride=1, bias=False),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.down_block(x)
        return x


class transition_up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(transition_up, self).__init__()
        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        x = self.up_block(x)
        return x


class DenseNetFCN(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out_init=48,
                 num_classes=2,
                 growth_rate=16,
                 layers=(4, 5, 7, 10, 12),
                 bottleneck=True,
                 bottleneck_layer=15):
        super(DenseNetFCN, self).__init__()
        self.init_conv = nn.Conv3d(ch_in,
                                   ch_out_init,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.num_transitions = len(layers)
        self.layers = layers
        self.bottleneck = bottleneck

        self.dbs_down = []
        self.dbs_up = []
        self.tds = []
        self.tus = []
        self.down_filters_in = [ch_out_init]
        for num_layers in layers:
            self.db = dense_block(self.down_filters_in[-1], num_layers, growth_rate)
            self.dbs_down.append(self.db)
            db_out = self.down_filters_in[-1] + (num_layers * growth_rate)
            self.down_filters_in.append(db_out)
            self.td = transition_down(db_out)
            self.tds.append(self.td)
        self.db_bottleneck = dense_block(
            self.down_filters_in[-1],
            bottleneck_layer,
            growth_rate,
            dropout_rate=0.2,
            bottleneck=True,
            grow_nb_filters=False,
            return_concat_list=True)
        self.bottleneck_out = (growth_rate * bottleneck_layer) + self.down_filters_in[-1]
        self.up_filters_in = [(growth_rate * bottleneck_layer)]
        self.up_filters_out = [self.bottleneck_out]
        layers = list(layers)
        layers.sort(reverse=True)
        for i, num_layers in enumerate(layers):
            self.tu = transition_up(self.up_filters_in[-1], self.up_filters_in[-1])
            self.tus.append(self.tu)
            concat_filters = self.down_filters_in[-(i+1)]
            db_in = concat_filters + self.up_filters_in[-1]
            # self.up_filters_in.append(db_in)
            self.db = dense_block(db_in,
                             num_layers,
                             growth_rate,
                             dropout_rate=0.2,
                             bottleneck=False,
                             grow_nb_filters=False,
                             return_concat_list=True)

            self.dbs_up.append(self.db)
            db_out = num_layers * growth_rate
            self.up_filters_in.append(db_out)
            self.up_filters_out.append(db_in + db_out)
        self.final_conv = nn.Conv3d(self.up_filters_out[-1],
                                    num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)


    def forward(self, x):
        x = self.init_conv(x)
        concat_list = []
        for i in range(self.num_transitions):
            db = self.dbs_down[i]
            x = db(x)
            # print(f'denseblock exit shape: {x.shape}')

            concat_list.append(x)
            td = self.tds[i]
            x = td(x)
        x, feature_list = self.db_bottleneck(x)
        # print(f'bottleneck shape: {x.shape}')
        keep_features = torch.concat(feature_list[1:], dim=1)
        # reverse order of concat list
        concat_list = concat_list[::-1]
        for i in range(self.num_transitions):
            tu = self.tus[i]
            x = tu(keep_features)
            concat = concat_list[i]
            x = torch.concat((x, concat), dim=1)
            db = self.dbs_up[i]
            x, feature_list = db(x)
            # print(f'denseblock exit shape: {x.shape}')
            keep_features = torch.concat(feature_list[1:], dim=1)
        x = self.final_conv(x)
        return x


model = DenseNetFCN(2)
# model = model.to('cuda')

input = torch.rand(3, 2, 64, 64, 64)

output = model(input)