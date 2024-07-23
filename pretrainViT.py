from transformers import ViTModel
import torch
import torch.nn as nn

class ViTForWildfireSpread(nn.Module):
    def __init__(self, num_labels=1):
        super(ViTForWildfireSpread, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_labels, kernel_size=1)

    def forward(self, x):
        x = self.vit(pixel_values=x).last_hidden_state
        x = x.permute(0, 2, 1).reshape(x.shape[0], 768, 14, 14)
        x = self.upsample(x)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ViTForWildfireSpread().to(device)
