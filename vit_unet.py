import torch
from torch import nn
from vision_transformer import vit_small
from task1model import CNNencoder_gn, CNNencoder_ln, Concat_gn, Concat_ln, Conv2dReLU

class VitUNet(nn.Module):
    def __init__(self,
                 vit=vit_small(patch_size=8, num_classes=0), 
                 patch_size=8
                 ):
        """8x8 dino backbone UNet

        Parameters
        ----------
        vit : torch.model, optional
            DINO model, by default vit_small(patch_size=8, num_classes=0)
        patch_size : int, optional
            do not change this value, by default 8
        """
        super().__init__()
        self.vit = vit
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.vit.load_state_dict(state_dict, strict=True)
        
        self.patch_size = patch_size
        
        self.projection = Conv2dReLU(384, 128, kernel_size=3, padding=1, use_groupnorm=True)
        
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        
        self.concat1 = Concat_gn(256, 64)
        self.convup1 = CNNencoder_gn(64, 64)
        self.concat2 = Concat_gn(128, 32)
        self.convup2 = CNNencoder_gn(32, 32)
        self.concat3 = Concat_gn(64, 16)
        self.convup3 = CNNencoder_gn(16, 16)
        self.concat4 = Concat_ln(32, 23)
        self.convup4 = CNNencoder_ln(23, 23)

        self.Segmentation_head = nn.Conv2d(23, 23, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        with torch.no_grad():
            B, _nc, w, h = x.shape
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            v = self.vit.get_last_value(x).permute(0,2,1)
            B, C, T = v.shape
            v = v.reshape(B, C, w0, h0)
        v = self.projection(v)
        
        # (B, in_channel, 512, 768)
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        # (B, 16, 512, 768)
        p1 = self.pooling(c1)
        # (B, 16, 256, 384)
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        # (B, 16, 256, 384)
        p2 = self.pooling(c2)
        # (B, 32, 128, 192)
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        # (B, 32, 128, 192)
        p3 = self.pooling(c3)
        # (B, 64, 64, 96)
        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)

        # (B, 128, 64, 96)
        u1 = self.concat1(v, c4)
        u1 = self.convup1(u1)
        # (B, 64, 64, 96)
        u1 = self.upsample(u1)
        # (B, 64, 128, 192)
        u2 = self.concat2(u1, c3)
        u2 = self.convup2(u2)
        # (B, 32, 128, 192)
        u2 = self.upsample(u2)
        # (B, 32, 256, 384)
        u3 = self.concat3(u2, c2)
        u3 = self.convup3(u3)
        # (B, 16, 256, 384)
        u3 = self.upsample(u3)
        # (B, 16, 512, 768)
        u4 = self.concat4(u3, c1)
        u4 = self.convup4(u4)
        # (B, 23, 512, 768)
        out = self.Segmentation_head(u4)
        # (B, 23, 512, 768)
        
        return out
    
class VitUNet16(nn.Module):
    def __init__(self,
                 vit=vit_small(patch_size=16, num_classes=0), 
                 patch_size=16
                 ):
        """16x16 dino backbone UNet

        Parameters
        ----------
        vit : torch.model, optional
            DINO model, by default vit_small(patch_size=16, num_classes=0)
        patch_size : int, optional
            do not change this value, by default 16
        """
        super().__init__()
        self.vit = vit
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.vit.load_state_dict(state_dict, strict=True)
        
        self.patch_size = patch_size
        
        self.projection = Conv2dReLU(384, 256, kernel_size=3, padding=1, use_groupnorm=True)
        
        self.pooling = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.conv1_1 = CNNencoder_gn(3, 16)
        self.conv1_2 = CNNencoder_gn(16, 16)
        self.conv2_1 = CNNencoder_gn(16, 32)
        self.conv2_2 = CNNencoder_gn(32, 32)
        self.conv3_1 = CNNencoder_gn(32, 64)
        self.conv3_2 = CNNencoder_gn(64, 64)
        self.conv4_1 = CNNencoder_gn(64, 128)
        self.conv4_2 = CNNencoder_gn(128, 128)
        self.conv5_1 = CNNencoder_gn(128, 256)
        self.conv5_2 = CNNencoder_gn(256, 256)
        
        self.concat1 = Concat_gn(512, 128)
        self.convup1 = CNNencoder_gn(128, 128)
        self.concat2 = Concat_gn(256, 64)
        self.convup2 = CNNencoder_gn(64, 64)
        self.concat3 = Concat_gn(128, 32)
        self.convup3 = CNNencoder_gn(32, 32)
        self.concat4 = Concat_gn(64, 16)
        self.convup4 = CNNencoder_gn(16, 16)
        self.concat5 = Concat_ln(32, 23)
        self.convup5 = CNNencoder_ln(23, 23)

        self.Segmentation_head = nn.Conv2d(23, 23, kernel_size=1, stride=1, bias=False)
    
    def forward(self, x):
        with torch.no_grad():
            B, _nc, w, h = x.shape
            w0 = w // self.patch_size
            h0 = h // self.patch_size
            v = self.vit.get_last_value(x).permute(0,2,1)
            B, C, _T = v.shape
            v = v.reshape(B, C, w0, h0) 
        v = self.projection(v)
        
        # (B, in_channel, 512, 768)
        c1 = self.conv1_1(x)
        c1 = self.conv1_2(c1)
        # (B, 16, 512, 768)
        p1 = self.pooling(c1)
        # (B, 16, 256, 384)
        c2 = self.conv2_1(p1)
        c2 = self.conv2_2(c2)
        # (B, 16, 256, 384)
        p2 = self.pooling(c2)
        # (B, 32, 128, 192)
        c3 = self.conv3_1(p2)
        c3 = self.conv3_2(c3)
        # (B, 32, 128, 192)
        p3 = self.pooling(c3)
        # (B, 64, 64, 96)
        c4 = self.conv4_1(p3)
        c4 = self.conv4_2(c4)
        # (B, 128, 64, 96)
        p4 = self.pooling(c4)
        # (B, 128, 32, 48)
        c5 = self.conv5_1(p4)
        c5 = self.conv5_2(c5)
        # (B, 256, 32, 48)
        
        u1 = self.concat1(v, c5)
        u1 = self.convup1(u1)
        # (B, 128, 32, 48)
        u1 = self.upsample(u1)
        # (B, 128, 64, 96)
        u2 = self.concat2(u1, c4)
        u2 = self.convup2(u2)
        # (B, 64, 64, 96)
        u2 = self.upsample(u2)
        # (B, 64, 128, 192)
        u3 = self.concat3(u2, c3)
        u3 = self.convup3(u3)
        # (B, 32, 128, 192)
        u3 = self.upsample(u3)
        # (B, 32, 256, 384)
        u4 = self.concat4(u3, c2)
        u4 = self.convup4(u4)
        # (B, 16, 256, 384)
        u4 = self.upsample(u4)
        # (B, 16, 512, 768)
        u5 = self.concat5(u4, c1)
        u5 = self.convup5(u5)
        # (B, 23, 512, 768)
        out = self.Segmentation_head(u5)
        # (B, 23, 512, 768)
        
        return out