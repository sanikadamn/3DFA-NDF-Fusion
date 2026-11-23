"""One actual baseline-config optimizer step; verify frozen NDF and inference.

Use SMOKE_ONLY=1 bash scripts/rlbench/train_peract_ndf.sh. Writes no policy
checkpoint; logs are isolated under the experiment's smoke directory.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from main import parse_arguments
from datasets import fetch_dataset_class
from modeling.policy import fetch_model_class
from modeling.encoder.text import fetch_tokenizers
from utils.trainers import fetch_train_tester
from utils.schedulers import fetch_scheduler


def main():
    args=parse_arguments()
    args.local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group('nccl')
    try:
        args.log_dir=args.base_log_dir/args.exp_log_dir/'smoke'
        args.log_dir.mkdir(parents=True,exist_ok=True)
        args.checkpoint=None  # this diagnostic never resumes/overwrites a training run
        tester=fetch_train_tester(args.dataset)(args,fetch_dataset_class(args.dataset),fetch_model_class(args.model_type))
        train,_,_=tester.get_loaders()
        if not len(train):
            raise ValueError('Not enough data for a full training batch')
        model=tester.get_model().cuda()
        model.workspace_normalizer.copy_(tester.get_workspace_normalizer().cuda())
        tester.tokenizer=fetch_tokenizers(args.backbone)
        ndf=model.encoder.ndf_fusion
        if ndf is None:
            raise ValueError('Run this diagnostic in MODE=ndf')
        before={k:v.detach().cpu().clone() for k,v in ndf.field.state_dict().items()}
        fusion_before=ndf.fusion.weight.detach().clone()
        optimizer=tester.get_optimizer(model)
        frozen_ids={id(p) for p in ndf.field.parameters()}
        assert not any(id(p) in frozen_ids for g in optimizer.param_groups for p in g['params'])
        model=DistributedDataParallel(model,device_ids=[args.local_rank],broadcast_buffers=False,find_unused_parameters=True)
        model.train()
        batch=next(iter(train))
        scheduler=fetch_scheduler(args.lr_scheduler,optimizer,args.train_iters)
        tester.train_one_step(model,optimizer,torch.GradScaler(),scheduler,batch)
        assert all(p.grad is None for p in ndf.field.parameters())
        assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in ndf.field.state_dict().items())
        assert torch.isfinite(ndf.fusion.weight).all()
        assert not torch.equal(ndf.fusion.weight,fusion_before), 'Fusion weights did not update'
        model.eval()
        with torch.inference_mode():
            action=tester._model_forward(model,batch,training=False)
        assert action.shape[-2:]==(1,8) and torch.isfinite(action).all()
        if dist.get_rank()==0:
            print('PASS: real batch backward/optimizer/inference; frozen NDF unchanged; fusion updated.')
            print('Action shape:',tuple(action.shape))
    finally:
        dist.destroy_process_group()


if __name__=='__main__':
    main()
