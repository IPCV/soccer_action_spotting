from typing import List

import torch
import torch.nn as nn

from IO import FeatureStream
from models import PlayerGraph
from .netvlad import NetVLAD, NetRVLAD


class StreamPooling(nn.Module):
    OUTPUT_SIZE = 512

    def __init__(self, input_size=512, pool="NetVLAD", frames_per_window=30, vocab_size=64):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super().__init__()
        self.pool = pool
        self.frames_per_window = frames_per_window
        self.vlad_k = vocab_size
        self.input_size = input_size

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(self.frames_per_window, stride=1)

        elif self.pool == "MAX++":
            self.pool_layer_before = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(2 * self.input_size, StreamPooling.OUTPUT_SIZE)

        elif self.pool == "AVG":
            self.pool_layer = nn.AvgPool1d(self.frames_per_window, stride=1)

        elif self.pool == "AVG++":
            self.pool_layer_before = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(2 * self.input_size, StreamPooling.OUTPUT_SIZE)

        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k,
                                      feature_size=self.input_size,
                                      add_batch_norm=True)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(self.input_size * self.vlad_k, StreamPooling.OUTPUT_SIZE)

        elif self.pool == "NetVLAD++":
            self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.input_size,
                                             add_batch_norm=True)
            self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                            feature_size=self.input_size,
                                            add_batch_norm=True)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(self.input_size * self.vlad_k, StreamPooling.OUTPUT_SIZE)

        elif self.pool == "NetRVLAD":
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k,
                                       feature_size=self.input_size,
                                       add_batch_norm=True)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(self.input_size * self.vlad_k, StreamPooling.OUTPUT_SIZE)

        elif self.pool == "NetRVLAD++":
            self.pool_layer_before = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                              feature_size=self.input_size,
                                              add_batch_norm=True)

            self.pool_layer_after = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.input_size,
                                             add_batch_norm=True)
            self.drop = nn.Dropout(p=0.4)
            self.fc = nn.Linear(self.input_size * self.vlad_k, StreamPooling.OUTPUT_SIZE)

    def forward(self, x):

        # Temporal pooling operation
        if self.pool == "MAX" or self.pool == "AVG":
            x = self.pool_layer(x.permute((0, 2, 1))).squeeze(-1)

        elif self.pool == "MAX++" or self.pool == "AVG++":
            half_frames = x.shape[1] // 2
            x_before, x_after = x[:, :half_frames, :], x[:, half_frames:, :]
            x_before = self.pool_layer_before(x_before.permute((0, 2, 1))).squeeze(-1)
            x_after = self.pool_layer_after(x_after.permute((0, 2, 1))).squeeze(-1)
            x = torch.cat((x_before, x_after), dim=1)
            x = self.drop(x)
            x = self.fc(x)

        elif self.pool == "NetVLAD" or self.pool == "NetRVLAD":
            x = self.pool_layer(x)
            x = self.drop(x)
            x = self.fc(x)

        elif self.pool == "NetVLAD++" or self.pool == "NetRVLAD++":
            half_frames = x.shape[1] // 2
            x_before = self.pool_layer_before(x[:, :half_frames, :])
            x_after = self.pool_layer_after(x[:, half_frames:, :])
            x = torch.cat((x_before, x_after), dim=1)
            x = self.drop(x)
            x = self.fc(x)
        return x


class Stream(StreamPooling):
    INPUT_SIZE = 512

    def __init__(self, input_size=512, pool="NetVLAD", frames_per_window=30, vocab_size=64):
        super().__init__(Stream.INPUT_SIZE, pool, frames_per_window, vocab_size)

        # are feature already PCA'ed?
        if input_size != Stream.INPUT_SIZE:
            self.feature_extractor = nn.Linear(input_size, Stream.INPUT_SIZE)

    def forward(self, x):
        # input_shape: (batch,frames,dim_features)
        batch_size, frame_rate, ndim = x.shape
        if ndim != Stream.INPUT_SIZE:
            x = x.reshape(batch_size * frame_rate, ndim)
            x = self.feature_extractor(x)
            x = x.reshape(batch_size, frame_rate, -1)

        x = StreamPooling.forward(self, x)
        return x


class GraphStream(PlayerGraph, StreamPooling):
    def __init__(self, gcn_backbone="GCN", pool="NetVLAD", frames_per_window=30, feature_multiplier=1,
                 use_calibration_confidence=False, feature_vector_size=8, vocab_size=64):
        PlayerGraph.__init__(self, gcn_backbone, frames_per_window, feature_multiplier, use_calibration_confidence,
                             feature_vector_size)
        StreamPooling.__init__(self, self.GRAPH_OUTPUT_SIZE, pool, frames_per_window, vocab_size)

    def forward(self, x):
        x = PlayerGraph.forward(self, x)
        x = StreamPooling.forward(self, x)
        return x


class Multistream(nn.Module):

    def __init__(self, streams: List, pool="NetVLAD", window_size_sec=15, frame_rate=2, num_classes=17,
                 vocab_size=64, gcn_backbone="GCN", feature_multiplier=1, use_calibration_confidence=False,
                 feature_vector_size=8, weights=None):

        """
        INPUTS: Two tensors of shape (batch_size,window_size,feature_size)
                corresponding to RGB and Audio features correspondingly
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Multistream, self).__init__()

        if not streams:
            raise ValueError('One or more streams must be defined')

        self.stream_names = [s.name for s in streams]
        self.num_streams = len(streams)

        self.pool = pool
        self.frames_per_window = window_size_sec * frame_rate
        self.num_classes = num_classes
        self.frame_rate = frame_rate
        self.vlad_k = vocab_size

        fc_input_size = 0
        for s in streams:
            if isinstance(s, FeatureStream):
                if s.name != 'graph':
                    setattr(self, s.name, Stream(s.features.dim, pool, self.frames_per_window, self.vlad_k))
                    fc_input_size += Stream.OUTPUT_SIZE
                else:
                    setattr(self, s.name, nn.BatchNorm1d(s.features.dim))
                    fc_input_size += s.features.dim
            else:
                setattr(self, s.name, GraphStream(gcn_backbone, pool, self.frames_per_window, feature_multiplier,
                                                  use_calibration_confidence, feature_vector_size, self.vlad_k))
                fc_input_size += GraphStream.OUTPUT_SIZE

        self.drop = nn.Dropout(p=0.4)
        self.fc = nn.Linear(fc_input_size, self.num_classes + 1)
        self.bn = nn.BatchNorm1d(self.num_classes + 1)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if weights:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")

    def forward(self, inputs):
        if self.num_streams > 1:
            x = tuple([getattr(self, name)(x) for name, x in zip(self.stream_names, inputs)])
            x = torch.cat(x, dim=1)
        else:
            x = getattr(self, self.stream_names[0])(inputs)

        x = self.drop(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.sigm(x)
        return x


if __name__ == "__main__":
    input_size, window_size_sec, frame_rate = 512, 15, 2
    model = Multistream([('audio', 512), ('rgb', 512)], "NetVLAD++", window_size_sec, frame_rate)
    print(model)
    audio = torch.rand([10, window_size_sec * frame_rate, 512])
    rgb = torch.rand([10, window_size_sec * frame_rate, 512])
    print(audio.shape)
    print(rgb.shape)
    output = model((audio, rgb))
    print(output.shape)
