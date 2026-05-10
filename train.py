import os
import sys  
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from model.gemdepth import GemDepth
from dataset.dataset_mix import DepthVideoDataset,safe_collate
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs
from loss.videoloss import *
from torch.utils.tensorboard import SummaryWriter

@hydra.main(version_base=None, config_path='config', config_name='stage1.yaml')
def main(cfg):
    set_seed(cfg.training.seed)
    Path(cfg.training.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True),  kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
    accelerator.init_trackers(project_name=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True))
    dataset_train = DepthVideoDataset(**cfg.dataset.train)
    train_loader = DataLoader(dataset=dataset_train,batch_size=cfg.dataloader.batch_size // cfg.num_gpus ,pin_memory=True, shuffle=True, num_workers=int(8), drop_last=True,collate_fn=safe_collate,timeout=3600)
    #load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = GemDepth(**model_configs[cfg.encoder]).to(accelerator.device)
    checkpoint = torch.load(cfg.model.video_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    model.pretrained.requires_grad_(False)
    dec_blocks_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('spatial_blocks') or  name.startswith('time_blocks') :
            dec_blocks_params.append(param)
        else:
            other_params.append(param)
            
    if cfg.optimizer.kind == 'adam':
        optim = torch.optim.Adam
    elif cfg.optimizer.kind == 'adamw':
        optim = torch.optim.AdamW
    else:
        print('Optimizer error')
        sys.exit(0)

    optimizer = optim(
    [
        {'params': dec_blocks_params, 'lr': 1e-5},  
        {'params': other_params, 'lr': 1e-6},      
    ],
    weight_decay=0.01  
)
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i}: lr = {param_group['lr']}")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, [1e-5,1e-6], cfg.total_step+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    train_loader, model, optimizer, lr_scheduler= accelerator.prepare(train_loader, model, optimizer, scheduler)
    model.to(accelerator.device)
    invariant_loss_func = VideoDepthLoss(pose_flag = cfg.pose_flag)
    total_step = 0
    should_keep_training = True
    writer = SummaryWriter(log_dir="./logs/train") 
    while should_keep_training:
        model.train()
        for data in tqdm(train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            if data is None:
                continue 
            image = data['image']
            depth_gt = data['depth']
            mask = (depth_gt>0).float()
            intrinsic_gt=data['IntM']
            extrinsic_gt=data['poses'] 
            with accelerator.autocast():
                depth_pred,pose_enc_list,extrinsic_pred,intrinsic_pred= model(image)
            loss_dict=invariant_loss_func(depth_pred.squeeze(2), depth_gt.squeeze(2),mask.squeeze(2),intrinsic_gt,extrinsic_gt,pose_enc_list,extrinsic_pred) 
            loss=loss_dict['total_loss']
            torch.cuda.empty_cache()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            if accelerator.is_main_process:   
                writer.add_scalar('train/loss', loss_dict['total_loss'], total_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], total_step)
                used_memory_MB = torch.cuda.memory_allocated() / 1024 / 1024
                max_used_memory_MB = torch.cuda.max_memory_allocated() / 1024 / 1024
                writer.add_scalar('train/memory_MB', used_memory_MB, total_step)
                writer.add_scalar('train/max_memory_MB', max_used_memory_MB, total_step)
            del loss
            del loss_dict
            torch.cuda.empty_cache()

            if total_step == cfg.total_step:
                should_keep_training = False
                break
            torch.cuda.empty_cache()
    if accelerator.is_main_process:
        save_path = Path(cfg.training.checkpoint_dir + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save

if __name__ == '__main__':
    main()