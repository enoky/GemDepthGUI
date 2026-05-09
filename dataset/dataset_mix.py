import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import re
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import Compose
from model.util.transform import Resize, NormalizeImage, PrepareForNet

def safe_collate(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.default_collate(batch)

class StatefulRandomCrop(A.RandomCrop):
    def __init__(self, height, width, **kwargs):
        super().__init__(height, width, **kwargs)
        self.last_crop_coords = (0, 0)

    def get_params_dependent_on_data(self, params, data):
        params_dict = super().get_params_dependent_on_data(params, data)
        x_min, y_min, x_max, y_max = params_dict["crop_coords"]
        self.last_crop_coords = (x_min, y_min)
        return params_dict
    
class RandomScale:
    def __init__(self, scale_limit, last_ch):
        self.scale_limit = scale_limit
        self.last_ch = last_ch

    def __call__(self, x):
        scale = np.random.uniform(*self.scale_limit)
        x = cv2.resize(x, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        x[..., -self.last_ch:] = x[..., -self.last_ch:] * scale
        return x
    
class RandomHorizontalFlip:
    def __init__(self, last_ch):
        self.last_ch = last_ch
    def __call__(self, x):
        if np.random.rand() > 0.5:
            x = cv2.flip(x, 1)
            seq_len = self.last_ch // 4
            x[..., -self.last_ch:-self.last_ch+seq_len] = -x[..., -self.last_ch:-self.last_ch+seq_len]
            x[..., -self.last_ch+2*seq_len:-self.last_ch+3*seq_len] = -x[..., -self.last_ch+2*seq_len:-self.last_ch+3*seq_len]
        return x
    
class RandomCropWithInfo(A.DualTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(RandomCropWithInfo, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.last_crop_coords = None

    def apply(self, img, x_min=0, y_min=0, **params):
        self.last_crop_coords = (x_min, y_min)
        return img[y_min:y_min + self.height, x_min:x_min + self.width]

class DepthVideoDataset(Dataset):
    def __init__(self, mode, data_dirs=[''], crop_size=518, seq_len=4):
        if data_dirs is None:
            data_dirs = ['']
        elif isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        self.mode = mode
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.tartanair_ratio=1 #30.5W
        self.vkitti_ratio=15 #2.1W
        self.max_depth_outer=200
        self.max_depth_inner = 80
        self.data_paths = []
        self.vkitti_data_paths=[]
        self.tartanair_data_paths=[]
        print(data_dirs)
        for data_dir in data_dirs:
            if 'vkitti' in data_dir:
                print("vkitti")
                depth_paths=sorted(glob.glob(data_dir + 'vkitti_1.3.1_depthgt'+'/*'))
                depth_paths = [path for path in depth_paths if os.path.isdir(path)]
                image_paths=sorted(glob.glob(data_dir + 'vkitti_1.3.1_rgb'+'/*'))
                image_paths = [path for path in image_paths if os.path.isdir(path)]
                pose_dir = os.path.join(data_dir, 'vkitti_1.3.1_extrinsicsgt')
                assert len(depth_paths)==len(image_paths)
                scenes=['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','fog','overcast','sunset','rain','morning']
                for scene in scenes:
                    pose_paths = sorted(
                    [f for f in glob.glob(os.path.join(pose_dir, f'*{scene}*.txt'))],
                    key=lambda x: int(re.match(r'.*?(\d+)', os.path.basename(x)).group(1))
                                       )
                    for k in range(len(depth_paths)):
                        depth_names = sorted(os.listdir(os.path.join(depth_paths[k], scene)))
                        image_names = sorted([file for file in os.listdir(os.path.join(image_paths[k], scene)) if file.endswith('.png')])
                        pose_names = pose_paths[k]
                        assert len(image_names) == len(depth_names)
                        image_num = len(image_names)
                        seq_num = image_num -seq_len + 1   
                        if mode == 'train':
                            start_idx = 0
                            end_idx = round(seq_num )
                        else:
                            start_idx = round(seq_num * 0.9)+1
                            end_idx = seq_num
                        for i in range(start_idx, end_idx):
                            set_paths = []
                            for j in range(seq_len):
                                image_path = os.path.join(image_paths[k], scene, image_names[i + j])
                                depth_path = os.path.join(depth_paths[k], scene, depth_names[i + j])
                                set_paths.append([image_path, depth_path,pose_names])
                            self.vkitti_data_paths.append(['vkitti', set_paths])     

            if 'tartanair' in data_dir:
                print("tartanair_true")
                scene_paths = sorted(glob.glob(data_dir + '/*/*/*'))
                for scene_path in scene_paths:
                    image_names = sorted([f for f in os.listdir(os.path.join(scene_path, 'image_left')) if f.endswith('.png')])
                    depth_names = sorted([f for f in os.listdir(os.path.join(scene_path, 'depth_left')) if f.endswith('.npy')])
                    pose_path = os.path.join(scene_path, 'pose_left.txt')
                    assert len(image_names) == len(depth_names)
                    image_num = len(image_names)
                    seq_num = image_num - seq_len + 1
                    poses = np.loadtxt(pose_path, delimiter=' ')
                    poses = poses[:, [1, 2, 0, 4, 5, 3, 6]]
                    if mode == 'train':
                        start_idx = 0
                        end_idx = round(seq_num )
                    else:
                        start_idx = round(seq_num * 0.9)+1
                        end_idx = seq_num
                    for i in range(start_idx, end_idx):
                        set_paths = []
                        for j in range(seq_len):
                            image_path = os.path.join(scene_path, 'image_left', image_names[i  + j])
                            depth_path = os.path.join(scene_path, 'depth_left', depth_names[i  + j])
                            pose_path=poses[i+j]
                            set_paths.append([image_path, depth_path,pose_path])
                        self.tartanair_data_paths.append(['TartanAir', set_paths])

        self.data_paths = self.vkitti_data_paths * self.vkitti_ratio + self.tartanair_data_paths*self.tartanair_ratio  

        self.scale = {
            'vkitti': RandomScale(scale_limit=(0.8, 0.85), last_ch=4*(seq_len-1)),
            'TartanAir': RandomScale(scale_limit=(0.8, 0.85), last_ch=4*(seq_len-1)),
        }

        self.flip = {
            'vkitti': RandomHorizontalFlip(last_ch=4*(seq_len-1)),
            'TartanAir': RandomHorizontalFlip(last_ch=4*(seq_len-1)),
        }
        
        self.transform = {
        'vkitti': A.Compose([  
        StatefulRandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.ToFloat()
        ]),
        'TartanAir': A.Compose([ 
        StatefulRandomCrop(height=crop_size, width=crop_size, p=1.0),
        A.ToFloat()
        ])
                        }
        
        
        self.transform_infer = Compose([
            Resize(
                width=crop_size,
                height=crop_size,
                resize_target=True ,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] )

    def __getitem__(self, item):
        while True:
            label, set_paths = self.data_paths[item]
            images = []
            images_ori=[]
            depths = []
            masks =[]
            poses = []
            path=[]
            if label in ['vkitti','TartanAir']:
                for image_path, depth_path,pose in set_paths:
                    image = cv2.imread(image_path).astype(np.float32) / 255
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_ori = (image - 0.5) * 2
                    if label == 'vkitti':
                        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)/100
                        depth = np.expand_dims(depth, axis=-1)
                        depth = depth.astype(np.float32)
                        mask=depth >0
                        depth[depth > self.max_depth_outer]=self.max_depth_outer
                        filename = os.path.basename(image_path)         
                        target_line_idx = int(os.path.splitext(filename)[0]) 
                        with open(pose, 'r') as f:
                            next(f)  
                            for i, line in enumerate(f):
                                if i == target_line_idx:
                                    values = list(map(float, line.split()))[1:]  
                                    rotation_translation = np.array(values[:12]).reshape(3, 4)
                                    homogeneous = np.array(values[12:])
                                    pose = np.vstack([rotation_translation, homogeneous.reshape(1, 4)])
                                    pose = pose.astype(np.float32)
                                    break  
                    if label == 'TartanAir':
                        depth = np.load(depth_path).astype(np.float32)[..., None]
                        mask=depth >0
                        depth[depth > self.max_depth_outer]=self.max_depth_outer
                        q = pose[3:]
                        t = pose[:3]
                        rotation = R.from_quat(q)
                        R_matrix = rotation.as_matrix()
                        T = np.eye(4)
                        T[:3, :3] = R_matrix
                        T[:3, 3] = t 
                        pose = T.astype(np.float32)  
                        pose=np.linalg.inv(pose)                        
                    path.append(depth_path)
                    poses.append(pose) 
                    sample = self.transform_infer({'image': image, 'depth': depth,'mask':mask,'image_ori':image_ori})
                    sample["image"]=np.transpose(sample["image"], (1, 2, 0))
                    images.append(sample["image"])
                    depths.append(np.expand_dims(sample['depth'], axis=-1))
                    masks.append(np.expand_dims(sample['mask'], axis=-1))
                    sample["image_ori"]=np.transpose(sample["image_ori"], (1, 2, 0))
                    images_ori.append(image)

            images = np.concatenate(images, axis=-1)  # H, W, 3T 
            images_ori = np.concatenate(images_ori, axis=-1)
            depths = np.concatenate(depths, axis=-1)  # H, W, T 
            masks = np.concatenate(masks, axis=-1)
            H,W,_=images_ori.shape
            h_new,w_new,_=images.shape
            factor=h_new/H
            all = np.concatenate((images,depths,masks), axis=-1)
            all = self.transform[label](image=all)['image']
            crop_transform = self.transform[label].transforms[0]
            left_margin, top_margin = crop_transform.last_crop_coords
            start = 0; end = 3 * self.seq_len; images = all[..., start:end]  # H, W, 3T
            start = end; end = start+self.seq_len; depths = all[..., start:end]  # H, W, T
            start = end; end = start + self.seq_len; masks = all[..., start:end]
            images = np.stack(np.split(images, self.seq_len, axis=-1), axis=0)#T H W 3
            depths = np.stack(np.split(depths, self.seq_len, axis=-1), axis=0)
            masks = np.stack(np.split(masks, self.seq_len, axis=-1), axis=0)
            images = torch.from_numpy(images).permute(0, 3, 1, 2)#T 3 H W
            depths = torch.from_numpy(depths).permute(0, 3, 1, 2)
            masks = torch.from_numpy(masks).permute(0, 3, 1, 2)
            if  label in ['vkitti','TartanAir']:
                inputs={}
                if label == 'TartanAir':
                    fx=320
                    fy=320
                    cx=320
                    cy=240
                if label == 'vkitti':
                    fx=725
                    fy=725
                    cx=620.5
                    cy=187        
                IntM = np.zeros((3, 3))
                IntM[2, 2] = 1.
                IntM[0, 0] = fx*factor
                IntM[1, 1] = fy*factor
                IntM[0, 2] = cx*factor-left_margin
                IntM[1, 2] = cy*factor-top_margin
                IntM = IntM.astype(np.float32)
                inputs = self.get_K(IntM, inputs)
                inv_K=inputs[('inv_K_pool', 0)]    
            sample = {
                'image': images,
                'depth': depths,
                'mask':masks,
                'label': label,
                'inv_K':inv_K,
                'poses':poses,
                'IntM':IntM,
                'path':path
            }
            return sample

    def __len__(self):
        return len(self.data_paths) // 4 * 4

    def get_K(self, K, inputs):
        inv_K = np.linalg.inv(K)
        K_pool = {}
        ho, wo = self.crop_size, self.crop_size
        for i in range(6):
            K_pool[(ho // 2**i, wo // 2**i)] = K.copy().astype('float32')
            K_pool[(ho // 2**i, wo // 2**i)][:2, :] /= 2**i

        inputs['K_pool'] = K_pool

        inputs[("inv_K_pool", 0)] = {}
        for k, v in K_pool.items():
            K44 = np.eye(4)
            K44[:3, :3] = v
            inputs[("inv_K_pool", 0)][k] = np.linalg.inv(K44).astype('float32')

        inputs[("inv_K", 0)] = torch.from_numpy(inv_K.astype('float32'))

        inputs[("K", 0)] = torch.from_numpy(K.astype('float32'))
    
        return inputs
    
    
if __name__ == '__main__':
    dataset = DepthVideoDataset('train',
                                data_dirs=["/mnt/data-a808/R11031/dynamic/"],
                                seq_len=32)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4,shuffle=True,pin_memory=True)
    with torch.no_grad():
        for i, sample_batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            path=sample_batch['path']
            images = sample_batch['image'].cuda()
            depths = sample_batch['depth'].cuda()
            masks = sample_batch['mask'].cuda()
            inv_K = sample_batch['inv_K']
            poses = sample_batch['poses']
        
        