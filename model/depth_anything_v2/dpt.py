import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet
import numpy as np
from tqdm import tqdm
def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
        
                    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        mode=False
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, mode,size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, mode,size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn,mode, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn,mode)
        
        out = self.scratch.output_conv1(path_1)
        if mode:
            out = F.interpolate(out.to(torch.float32), 
                    (int(patch_h * 14), int(patch_w * 14)), 
                    mode="bilinear", align_corners=True)
            out = out.to(torch.bfloat16)
        else:
            out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out

INFER_LEN = 3
OVERLAP = 1
class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        features=256, 
        out_channels=[256, 512, 1024, 1024], 
        use_bn=False, 
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        # self.init_weights()
    def init_weights(self):         
        for m in self.depth_head.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.trunc_normal_(m.weight, std=0.02)  
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        B,T,C,H,W=x.shape
        x=x.flatten(0,1)
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.relu(depth)
        
        return depth.unflatten(0, (B, T))
    def infer_video_depth(self, frames, target_fps=1):
        # frame_height, frame_width = frames[0].shape[:-2]
        T,C,H,W=frames.shape
        frame_list = [frames[i] for i in range(frames.shape[0])]
        frame_step = INFER_LEN - OVERLAP
        org_video_len = len(frame_list)
        append_frame_len = (frame_step - (org_video_len % frame_step)) % frame_step + (INFER_LEN - frame_step)
        frame_list = frame_list + [frame_list[-1].clone()] * append_frame_len
        output_path='/data2/lyc/outputs/videos'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = None 
        depth_list = []
        pre_input = None
        seq_len=3
        overlap=1
        prev_frames=None
        prev_end=0
        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            cur_list = []
            for i in range(INFER_LEN):
                images=frame_list[frame_id+i].reshape(1,C,H,W)
                cur_list.append(images)
            cur_input = torch.stack(cur_list, dim=1)
            with torch.no_grad():
                depth_pred = self.forward(cur_input) 
            B,T,_,H,W=depth_pred.shape
            predictions = np.zeros((B, T, H, W)).astype(np.float32)  
            count = np.zeros((B, T))  
            for b in range(B):                
                frame = depth_pred[b, 0:seq_len].cpu().numpy()  # (T, C, H, W)
                frame = frame.transpose(0, 2, 3, 1)  #  (T, H, W, C)
                predictions[b, 0:0+seq_len] += frame.squeeze(-1)
                count[b, 0:0+seq_len] += 1
            predictions /= count[:, :, None, None]
            if prev_frames is None:
                prev_frames = predictions  
                prev_end = seq_len  
            else:
                overlap_frames_prev = prev_frames[:, -overlap:]  
                overlap_frames_curr = predictions[:, :overlap] 
                avg_overlap_frames = (overlap_frames_prev + overlap_frames_curr) / 2
                
                non_overlap_frames = predictions[:, overlap:]  
                prev_frames = np.concatenate([prev_frames[:, :prev_frames.shape[1]-overlap], avg_overlap_frames, non_overlap_frames], axis=1)
                prev_end = prev_end + seq_len - overlap

        # output_file = f"{output_path}/final_video_step.mp4"
        # video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (W, H))  
        # for b in range(B):
        #     for t in range(prev_frames.shape[1]):  
        #         frame = prev_frames[b, t]  
        #         frame =((frame - frame.min()) / (frame.max() - frame.min()))
        #         cmapper = matplotlib.cm.get_cmap('viridis')
        #         value1 = cmapper(frame, bytes=True)
        #         # frame = frame.astype(np.uint8)         
        #         # frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #         value1 = value1[:, :, :3]
        #         video_writer.write(value1)
        # video_writer.release()
        return prev_frames[:, :-1, :, :].reshape(B, 390, 1, H, W)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        # image, (h, w) = self.image2tensor(raw_image, input_size)
        
        depth = self.forward(raw_image)
        
        # depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        
        return depth
    
    def image2tensor(self, raw_image, input_size=518):    
        raw_image=raw_image.flatten(0,1)
        if isinstance(raw_image, torch.Tensor):
            raw_image = raw_image.cpu().numpy()  
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        b, c ,h, w = raw_image.shape
        
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({'image': raw_image})['image']
        image = torch.from_numpy(image)
        
        # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # image = image.to(DEVICE)
        
        return image, (h, w)
