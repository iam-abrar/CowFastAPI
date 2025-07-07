# model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class FiLMBlock(nn.Module):
    def __init__(self, tab_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(tab_dim, feat_dim)
        self.beta = nn.Linear(tab_dim, feat_dim)

    def forward(self, cnn_feat, tab_feat):
        gamma = self.gamma(tab_feat).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(tab_feat).unsqueeze(-1).unsqueeze(-1)
        return gamma * cnn_feat + beta

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, img_dim, tab_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(tab_dim, img_dim)
        self.key = nn.Linear(img_dim, img_dim)
        self.value = nn.Linear(img_dim, img_dim)
        self.out_proj = nn.Linear(img_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feat, tab_feat):
        Q = self.query(tab_feat).unsqueeze(1)
        K = self.key(img_feat).unsqueeze(1)
        V = self.value(img_feat).unsqueeze(1)
        dim = Q.size(-1)
        attn = torch.softmax((Q @ K.transpose(-2, -1)) / (dim ** 0.5), dim=-1)
        out = attn @ V
        out = self.out_proj(out.squeeze(1))
        out = self.dropout(out)
        return self.norm(out)

class CowWeightEstimator(nn.Module):
    def __init__(self, tabular_dim=8):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Extend to 4 channels
        orig_conv = self.cnn.conv1
        self.cnn.conv1 = nn.Conv2d(
            4, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )
        with torch.no_grad():
            self.cnn.conv1.weight[:, :3, :, :] = orig_conv.weight
            self.cnn.conv1.weight[:, 3:, :, :] = orig_conv.weight.mean(dim=1, keepdim=True)

        # Encoder layers
        self.layer1 = self.cnn.layer1
        self.layer2 = self.cnn.layer2
        self.layer3 = self.cnn.layer3
        self.layer4 = self.cnn.layer4
        self.num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # Tabular path
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # FiLM layers
        self.film2 = FiLMBlock(32, 128)
        self.film3 = FiLMBlock(32, 256)

        # Attention and gating
        self.attn_fusion = MultiHeadAttentionFusion(self.num_ftrs, 32)
        self.tabular_gate = nn.Sequential(
            nn.Linear(32, self.num_ftrs),
            nn.Sigmoid()
        )

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_ftrs),
            nn.Linear(self.num_ftrs, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, img_with_depth, tabular, use_cnn=True, use_tabular=True):
        tab_feat = self.tabular_mlp(tabular) if use_tabular else torch.zeros((img_with_depth.shape[0], 32), device=img_with_depth.device)

        if use_cnn:
            x = self.cnn.relu(self.cnn.bn1(self.cnn.conv1(img_with_depth)))
            x = self.cnn.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.film2(x, tab_feat)
            x = self.layer3(x)
            x = self.film3(x, tab_feat)
            x = self.layer4(x)
            x = self.cnn.avgpool(x)
            img_feat = torch.flatten(x, 1)
        else:
            img_feat = torch.zeros((tabular.shape[0], self.num_ftrs), device=tabular.device)

        fused = self.attn_fusion(img_feat, tab_feat)
        gate = self.tabular_gate(tab_feat)
        fused = fused * gate

        out = self.head(fused)
        return out.squeeze(1)
