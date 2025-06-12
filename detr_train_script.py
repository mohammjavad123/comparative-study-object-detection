
import os
import numpy as np
np.float = float  # Fix for deprecated numpy float in newer versions

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from detr.datasets import build_dataset
from detr.engine import train_one_epoch, evaluate
from detr.models.detr import build_detr
from detr.util import misc as utils

def main():
    # Training arguments
    args = {
        'batch_size': 2,
        'epochs': 1,
        'lr': 1e-5,
        'lr_backbone': 1e-6,
        'output_dir': '/home/mrajabi/vision/detr/output',
        'coco_path': '/home/mrajabi/vision/detr/data-set',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume': 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
    }

    # Set distributed mode
    utils.init_distributed_mode(args)

    print("Loading datasets...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args['batch_size'], drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=4, collate_fn=utils.collate_fn)
    data_loader_val = DataLoader(dataset_val, batch_size=args['batch_size'], sampler=sampler_val, drop_last=False, num_workers=4, collate_fn=utils.collate_fn)

    print("Building model...")
    model, criterion, postprocessors = build_detr(args)

    model.to(args['device'])

    print("Building optimizer...")
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args['lr_backbone'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args['lr'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)

    print("Starting training...")
    for epoch in range(args['epochs']):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, args['device'], epoch, 10, None
        )
        lr_scheduler.step()

        print("Evaluating...")
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, args['device']
        )

        if args['output_dir']:
            utils.save_on_master(model.state_dict(), os.path.join(args['output_dir'], f'checkpoint_{epoch:04}.pth'))

if __name__ == "__main__":
    main()
