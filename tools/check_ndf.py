import argparse
import json
import torch
from modeling.ndf.config import NDFConfig
from modeling.ndf.fusion import FrozenNDFFusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/ndf.json')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.manual_seed(42)
    model = FrozenNDFFusion(120, NDFConfig.read(args.config)).to(args.device).train()
    cloud = torch.randn(512,3,device=args.device)*.08
    queries = cloud[:32].unsqueeze(0)
    z,c = model.encode_cloud(cloud)
    before = model.describe(z,c,queries)
    r,_ = torch.linalg.qr(torch.randn(3,3,device=args.device))
    r[:,-1] *= torch.det(r)
    t = torch.tensor([.2,-.1,.3],device=args.device)
    z2,c2 = model.encode_cloud(cloud@r.T+t)
    after = model.describe(z2,c2,queries@r.T+t)
    torch.testing.assert_close(before,after,atol=3e-4,rtol=3e-3)
    visual = torch.randn(1,32,120,device=args.device,requires_grad=True)
    model.fusion(torch.cat((visual,before),-1)).square().mean().backward()
    assert not model.field.training
    assert all(not x.requires_grad and x.grad is None for x in model.field.parameters())
    assert model.fusion.weight.grad[:,120:].abs().sum()>0
    print(json.dumps({'descriptor_shape':list(before.shape), 'fusion_input_dim':model.fusion.in_features,
                      'max_descriptor_error':(before-after).abs().max().item(),
                      'frozen_field':True,'fusion_has_gradient':True},indent=2))


if __name__=='__main__':
    main()
