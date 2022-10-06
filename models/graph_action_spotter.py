from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv import EdgeConv, DynamicEdgeConv
from torch_geometric.nn.conv import GCNConv

from .netvlad import NetVLAD, NetRVLAD


class PlayerGraph(nn.Module):
    def __init__(self, gcn_backbone="GCN", frames_per_window=30, feature_multiplier=1, use_calibration_confidence=False,
                 feature_vector_size=8):
        super(PlayerGraph, self).__init__()

        self.gcn_backbone = gcn_backbone
        self.frames_per_window = frames_per_window
        self.GRAPH_OUTPUT_SIZE = 152 * feature_multiplier

        multiplier = 2 * feature_multiplier

        input_channel = feature_vector_size
        if use_calibration_confidence:
            input_channel += 1

        if self.gcn_backbone == "GCN":
            self.conv_1 = GCNConv(input_channel, 8 * multiplier)
            self.conv_2 = GCNConv(8 * multiplier, 16 * multiplier)
            self.conv_3 = GCNConv(16 * multiplier, 32 * multiplier)
            self.conv_4 = GCNConv(32 * multiplier, 76 * multiplier)

        elif self.gcn_backbone == "EdgeConvGCN":
            self.conv_1 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2 * input_channel, 8 * multiplier),
                                                         nn.BatchNorm1d(8 * multiplier)]))
            self.conv_2 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 8 * multiplier, 16 * multiplier),
                                                         nn.BatchNorm1d(16 * multiplier)]))
            self.conv_3 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 16 * multiplier, 32 * multiplier),
                                                         nn.BatchNorm1d(32 * multiplier)]))
            self.conv_4 = EdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 32 * multiplier, 76 * multiplier),
                                                         nn.BatchNorm1d(76 * multiplier)]))

        elif self.gcn_backbone == "DynamicEdgeConvGCN":
            self.conv_1 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2 * input_channel, 8 * multiplier),
                                                                nn.BatchNorm1d(8 * multiplier)]), k=3)
            self.conv_2 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 8 * multiplier, 16 * multiplier),
                                                                nn.BatchNorm1d(16 * multiplier)]), k=3)
            self.conv_3 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 16 * multiplier, 32 * multiplier),
                                                                nn.BatchNorm1d(32 * multiplier)]), k=3)
            self.conv_4 = DynamicEdgeConv(torch.nn.Sequential(*[nn.Linear(2 * 32 * multiplier, 76 * multiplier),
                                                                nn.BatchNorm1d(76 * multiplier)]), k=3)

        elif "resGCN" in self.gcn_backbone:
            output_channel = 76 * multiplier
            hidden_channels = 64
            self.num_layers = int(self.gcn_backbone.split("-")[-1])

            self.node_encoder = nn.Linear(input_channel, hidden_channels)
            self.edge_encoder = nn.Linear(input_channel, hidden_channels)
            self.layers = torch.nn.ModuleList()
            for i in range(1, self.num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=1.0, learn_t=True, num_layers=2,
                               norm='layer')
                norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
                act = nn.ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
                self.layers.append(layer)

            self.linear = nn.Linear(hidden_channels, output_channel)

    def forward(self, inputs):
        batch_size = ceil(inputs.num_graphs / self.frames_per_window)

        x, edge_index, batch = inputs.x, inputs.edge_index, inputs.batch
        if self.gcn_backbone == "GCN" or self.gcn_backbone == "EdgeConvGCN":
            x = F.relu(self.conv_1(x, edge_index))
            x = F.relu(self.conv_2(x, edge_index))
            x = F.relu(self.conv_3(x, edge_index))
            x = F.relu(self.conv_4(x, edge_index))
        elif "DynamicEdgeConvGCN" in self.gcn_backbone:
            x = F.relu(self.conv_1(x, batch))
            x = F.relu(self.conv_2(x, batch))
            x = F.relu(self.conv_3(x, batch))
            x = F.relu(self.conv_4(x, batch))
        elif "resGCN" in self.gcn_backbone:
            x = self.node_encoder(x)
            x = self.layers[0].conv(x, edge_index)
            for layer in self.layers[1:]:
                x = layer(x, edge_index)
            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            x = self.linear(x)
        x = global_max_pool(x, batch)

        # Zero padding incomplete batches
        expected_size = batch_size * self.frames_per_window
        x = torch.cat([x, torch.zeros(expected_size - x.shape[0], x.shape[1]).to(x.device)], 0)

        x = x.reshape(batch_size, self.frames_per_window, x.shape[1])
        return x


class GraphActionSpotter(PlayerGraph):

    def __init__(self, pool="NetVLAD", gcn_backbone="GCN", window_size_sec=15, frame_rate=2,
                 num_classes=17, vocab_size=64, feature_multiplier=1, use_calibration_confidence=False,
                 feature_vector_size=8, weights=None):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """
        super(GraphActionSpotter, self).__init__(gcn_backbone, window_size_sec*frame_rate, feature_multiplier,
                                                 use_calibration_confidence, feature_vector_size)

        self.num_classes = num_classes
        self.pool = pool
        self.vlad_k = vocab_size

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(self.frames_per_window, stride=1)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "MAX++":
            self.pool_layer_before = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.fc1 = nn.Linear(2 * self.GRAPH_OUTPUT_SIZE, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "AVG":
            self.pool_layer = nn.AvgPool1d(self.frames_per_window, stride=1)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "AVG++":
            self.pool_layer_before = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.fc1 = nn.Linear(2 * self.GRAPH_OUTPUT_SIZE, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k,
                                      feature_size=self.GRAPH_OUTPUT_SIZE,
                                      add_batch_norm=True)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE * self.vlad_k, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "NetVLAD++":
            self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.GRAPH_OUTPUT_SIZE,
                                             add_batch_norm=True)
            self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                            feature_size=self.GRAPH_OUTPUT_SIZE,
                                            add_batch_norm=True)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE * self.vlad_k, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "NetRVLAD":
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k,
                                       feature_size=self.GRAPH_OUTPUT_SIZE,
                                       add_batch_norm=True)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE * self.vlad_k, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        elif self.pool == "NetRVLAD++":
            self.pool_layer_before = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                              feature_size=self.GRAPH_OUTPUT_SIZE,
                                              add_batch_norm=True)

            self.pool_layer_after = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.GRAPH_OUTPUT_SIZE,
                                             add_batch_norm=True)
            self.fc1 = nn.Linear(self.GRAPH_OUTPUT_SIZE * self.vlad_k, 512)
            self.fc2 = nn.Linear(512, self.num_classes + 1)

        self.drop = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(self.num_classes+1)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if weights:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")

    def forward(self, x):

        x = PlayerGraph.forward(self, x)

        # Temporal pooling operation
        if self.pool == "MAX" or self.pool == "AVG":
            x = self.pool_layer(x.permute((0, 2, 1))).squeeze(-1)

        elif self.pool == "MAX++" or self.pool == "AVG++":
            half_frames = x.shape[1] // 2
            x_before = x[:, :half_frames, :]
            x_after = x[:, half_frames:, :]
            x_before_pooled = self.pool_layer_before(x_before.permute((0, 2, 1))).squeeze(-1)
            x_pooled = self.pool_layer_after(x_after.permute((0, 2, 1))).squeeze(-1)
            x = torch.cat((x_before_pooled, x_pooled), dim=1)

        elif self.pool == "NetVLAD" or self.pool == "NetRVLAD":
            x = self.pool_layer(x)

        elif self.pool == "NetVLAD++" or self.pool == "NetRVLAD++":
            half_frames = x.shape[1] // 2
            x_before_pooled = self.pool_layer_before(x[:, :half_frames, :])
            x_pooled = self.pool_layer_after(x[:, half_frames:, :])
            x = torch.cat((x_before_pooled, x_pooled), dim=1)

        x = self.drop(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigm(x)
        return x


if __name__ == "__main__":
    model = GraphActionSpotter()
    print(model)
