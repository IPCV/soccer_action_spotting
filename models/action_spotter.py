import torch
import torch.nn as nn

from .netvlad import NetVLAD, NetRVLAD


class ActionSpotter(nn.Module):

    def __init__(self, pool="NetVLAD", input_size=512, window_size_sec=15, frame_rate=2, num_classes=17, vocab_size=64,
                 weights=None):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(ActionSpotter, self).__init__()

        self.frames_per_window = window_size_sec * frame_rate
        self.input_size = input_size
        self.num_classes = num_classes
        self.frame_rate = frame_rate
        self.pool = pool
        self.vlad_k = vocab_size

        # are feature already PCA'ed?
        if self.input_size != 512:
            self.feature_extractor = nn.Linear(self.input_size, 512)
            self.input_size = 512

        if self.pool == "MAX":
            self.pool_layer = nn.MaxPool1d(self.frames_per_window, stride=1)
            self.fc = nn.Linear(self.input_size, self.num_classes + 1)

        elif self.pool == "MAX++":
            self.pool_layer_before = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.MaxPool1d(self.frames_per_window // 2, stride=1)
            self.fc = nn.Linear(2 * self.input_size, self.num_classes + 1)

        elif self.pool == "AVG":
            self.pool_layer = nn.AvgPool1d(self.frames_per_window, stride=1)
            self.fc = nn.Linear(self.input_size, self.num_classes + 1)

        elif self.pool == "AVG++":
            self.pool_layer_before = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.pool_layer_after = nn.AvgPool1d(self.frames_per_window // 2, stride=1)
            self.fc = nn.Linear(2 * self.input_size, self.num_classes + 1)

        elif self.pool == "NetVLAD":
            self.pool_layer = NetVLAD(cluster_size=self.vlad_k,
                                      feature_size=self.input_size,
                                      add_batch_norm=True)
            self.fc = nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)

        elif self.pool == "NetVLAD++":
            self.pool_layer_before = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.input_size,
                                             add_batch_norm=True)
            self.pool_layer_after = NetVLAD(cluster_size=int(self.vlad_k / 2),
                                            feature_size=self.input_size,
                                            add_batch_norm=True)
            self.fc = nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)

        elif self.pool == "NetRVLAD":
            self.pool_layer = NetRVLAD(cluster_size=self.vlad_k,
                                       feature_size=self.input_size,
                                       add_batch_norm=True)
            self.fc = nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)

        elif self.pool == "NetRVLAD++":
            self.pool_layer_before = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                              feature_size=self.input_size,
                                              add_batch_norm=True)

            self.pool_layer_after = NetRVLAD(cluster_size=int(self.vlad_k / 2),
                                             feature_size=self.input_size,
                                             add_batch_norm=True)
            self.fc = nn.Linear(self.input_size * self.vlad_k, self.num_classes + 1)

        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()

        self.load_weights(weights=weights)

    def load_weights(self, weights=None):
        if weights:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")

    def forward(self, x):
        # input_shape: (batch,frames,dim_features)

        batch_size, frames_per_window_, ndim = x.shape
        if ndim != self.input_size:
            x = x.reshape(batch_size * frames_per_window_, ndim)
            x = self.feature_extractor(x)
            x = x.reshape(batch_size, frames_per_window_, -1)

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
        x = self.fc(x)
        x = self.sigm(x)
        return x


if __name__ == "__main__":
    input_size, window_size_sec, frame_rate = 512, 15, 2
    model = ActionSpotter("NetVLAD", input_size, window_size_sec, frame_rate)
    print(model)
    inp = torch.rand([10, window_size_sec*frame_rate, 512])
    print(inp.shape)
    output = model(inp)
    print(output.shape)
