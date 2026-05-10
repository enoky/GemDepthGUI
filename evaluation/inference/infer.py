import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np
import sys  
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)  
if root_dir not in sys.path:
    sys.path.append(root_dir)
from model.gemdepth import GemDepth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='/mnt/sfs_turbo_new/R11031/video_test/')
    parser.add_argument('--json_file', type=str, default="/mnt/data-a808/R11031/kitti/kitti_video_500.json")
    parser.add_argument('--datasets', type=str, nargs='+', default=['kitti'])
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])

    args = parser.parse_args()
    for dataset in args.datasets:
        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        gemdepth = GemDepth(**model_configs[args.encoder])
        checkpoint = torch.load("./checkpoint/gemdepth.pth", map_location='cpu',weights_only=False)
        gemdepth.load_state_dict(checkpoint, strict=True)
        gemdepth = gemdepth.to(DEVICE).eval()
        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)
        for data in tqdm(json_data):
             for key in data.keys():
                value = data[key]
                infer_paths = []
                videos = []
                for images in value:
                    image_path = os.path.join(root_path, images['image'])
                    infer_path = (args.infer_path + '/'+ 'kitti_icml' +'/' + images['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    infer_paths.append(infer_path)
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    videos.append(img)
                videos = np.stack(videos, axis=0)
                target_fps=1
                depths, fps = gemdepth.infer_video_depth(videos, target_fps, input_size=args.input_size, device=DEVICE, fp32=True)
                for i in range(len(infer_paths)):
                    infer_path = infer_paths[i]
                    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
                    depth = depths[i]
                    np.save(infer_path, depth)
                    