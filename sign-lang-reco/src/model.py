import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTMAttn(nn.Module):
    def __init__(self, num_classes, hidden_size=256, dropout=0.2):
        super().__init__()

        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        except AttributeError:
            weights = None

        backbone = models.resnet34(weights=weights)
        backbone.fc = nn.Identity()
        self.cnn = backbone
        feat_dim = 512

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)          # (B * T, 512)
        feat = self.proj(feat)      # (B * T, 512)
        feat = feat.view(B, T, -1)  # (B, T, 512)

        out, _ = self.lstm(feat)    # (B, T, 2 * hidden)

        attn_logits = self.attn(out).squeeze(-1)   # (B, T)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)  # (B, T, 1)

        pooled = (out * attn_weights).sum(dim=1)    # (B, 2 * hidden)
        logits = self.head(pooled)

        return logits