import torch
import torch.nn as nn
from torchvision import models
from transformers import ViTModel, ViTConfig

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=2, cnn_model='convnext_tiny', vit_model='google/vit-base-patch16-224-in21k'):
        super().__init__()

        # CNN часть
        if cnn_model == 'convnext_tiny':
            self.cnn_backbone = models.convnext_tiny()
            cnn_out_features = 768
        else:
            raise ValueError(f"Unsupported CNN model: {cnn_model}")

        self.cnn_backbone.classifier = nn.Identity()

        # ViT часть
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3
        )
        self.vit = ViTModel(config)
        # self.vit = ViTModel().from_pretrained(vit_model)
        vit_out_features = self.vit.config.hidden_size

        # Проекционные слои
        self.projection_dim = 256
        self.cnn_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cnn_out_features, self.projection_dim),
            nn.GELU()
        )

        self.vit_proj = nn.Sequential(
            nn.Linear(vit_out_features, self.projection_dim),
            nn.GELU()
        )

        # Классификатор (ранее назывался classifier)
        self.fc = nn.Sequential(  # Переименовано в fc для совместимости
            nn.LayerNorm(self.projection_dim * 2),
            nn.Linear(self.projection_dim * 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.cnn_proj, self.vit_proj, self.fc]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        cnn_feats = self.cnn_proj(self.cnn_backbone(x))
        vit_feats = self.vit_proj(self.vit(x).last_hidden_state[:, 0, :])
        return self.fc(torch.cat([cnn_feats, vit_feats], dim=1))

    def freeze_cnn(self):
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

    def freeze_vit(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def change_num_classes(self, num_classes, device):
        num_features = self.fc[1].in_features
        self.fc = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_classes)
        ).to(device)

    @staticmethod
    def from_pretrained(pretrained_model_path, device, num_classes = 2):
        model = HybridCNNTransformer(num_classes=num_classes).to(device)
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        return model