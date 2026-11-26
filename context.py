"""Cached four-input context for Practical-RIFE v4.15. Original code, MIT."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import torch
from torch import nn
import torch.nn.functional as F


def warp(image, flow):
    """RIFE-compatible backward warp; standalone version for synthetic tests."""
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=flow.device),
                            torch.linspace(-1, 1, w, device=flow.device), indexing="ij")
    grid = torch.stack((xx, yy), -1)[None]
    d = torch.cat((flow[:, :1] / ((w - 1) / 2), flow[:, 1:] / ((h - 1) / 2)), 1)
    return F.grid_sample(image.float(), grid + d.permute(0, 2, 3, 1),
                         padding_mode="border", align_corners=True)


class Sep(nn.Sequential):
    def __init__(self, cin, cout, stride=1):
        super().__init__(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin),
                         nn.Conv2d(cin, cout, 1), nn.LeakyReLU(.2, inplace=True))


class ContextHead(nn.Module):
    def __init__(self, channels=16, widths=(32, 48, 64), flow_limit=8., residual_limit=.03):
        super().__init__()
        self.channels, self.widths = channels, tuple(widths)
        self.flow_limit, self.residual_limit = flow_limit, residual_limit
        self.encoder = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.LeakyReLU(.2),
                                     Sep(channels, channels))
        nf = channels + 3
        cin = 4 * nf + 4 + 1 + 2
        self.align = nn.Sequential(nn.Conv2d(cin, 24, 3, padding=1), nn.LeakyReLU(.2),
                                   Sep(24, 24), nn.Conv2d(24, 4, 1))
        nn.init.zeros_(self.align[-1].weight)
        nn.init.zeros_(self.align[-1].bias)
        a, b, c = widths
        self.e0 = nn.Sequential(nn.Conv2d(cin, a, 3, padding=1), nn.LeakyReLU(.2), Sep(a, a))
        self.e1 = nn.Sequential(Sep(a, b, 2), Sep(b, b))
        self.e2 = nn.Sequential(Sep(b, c, 2), Sep(c, c))
        self.d1, self.d0 = Sep(b + c, b), Sep(a + b, a)
        self.out = nn.Conv2d(a, 8, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def config(self):
        return dict(channels=self.channels, widths=self.widths, flow_limit=self.flow_limit,
                    residual_limit=self.residual_limit)

    def encode(self, rgb):
        rgb = F.avg_pool2d(rgb, 4)
        return torch.cat((self.encoder(rgb), rgb), 1)

    def forward(self, features, flow, logit, valid, warp_fn=warp):
        size = features[0].shape[-2:]
        fq = F.interpolate(flow, size, mode="bilinear", align_corners=False) * .25
        mq = F.interpolate(logit, size, mode="bilinear", align_corners=False).sigmoid()
        validity = []
        for v in valid:
            if not torch.is_tensor(v):
                v = fq.new_full((fq.shape[0], 1, 1, 1), float(v))
            validity.append(v.to(fq).expand(-1, 1, *size))
        vp, vn = validity
        # Missing context duplicates its central feature AND its sampling flow.
        # Extrapolating a duplicate by 3x would create false motion evidence.
        fs = [fq[:, :2] * (1 + 2 * vp), fq[:, :2], fq[:, 2:], fq[:, 2:] * (1 + 2 * vn)]
        aligned = [warp_fn(f.float(), d) for f, d in zip(features, fs)]
        guide = [fq / 8., mq, vp, vn]
        offsets = 4. * self.align(torch.cat((*aligned, *guide), 1)).tanh()
        aligned[0] = warp_fn(features[0].float(), fs[0] + offsets[:, :2] * vp)
        aligned[3] = warp_fn(features[3].float(), fs[3] + offsets[:, 2:] * vn)
        a = self.e0(torch.cat((*aligned, *guide), 1))
        b, c = self.e1(a), None
        c = self.e2(b)
        b = self.d1(torch.cat((b, F.interpolate(c, b.shape[-2:], mode="bilinear", align_corners=False)), 1))
        a = self.d0(torch.cat((a, F.interpolate(b, a.shape[-2:], mode="bilinear", align_corners=False)), 1))
        delta = F.interpolate(self.out(a), flow.shape[-2:], mode="bilinear", align_corners=False).float()
        return (self.flow_limit * delta[:, :4].tanh(), 2. * delta[:, 4:5].tanh(),
                self.residual_limit * delta[:, 5:8].tanh())


class ContextRIFE(nn.Module):
    """Frozen IFNet stages, context correction before the final full-size RGB warps."""
    def __init__(self, base, **config):
        super().__init__()
        self.base, self.context = base, ContextHead(**config)
        self.base.requires_grad_(False).eval()
        self.warp = getattr(sys.modules.get(base.__class__.__module__), "warp", warp)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, timestep=.5, scale_list=None, *, outer=None,
                features=None, valid=(1., 1.), **kwargs):
        if timestep != .5 or kwargs.get("ensemble", False):
            raise ValueError("Track C supports midpoint, single-pass v4.15 only")
        scales = [8., 4., 2., 1.] if scale_list is None else scale_list
        i0, i1 = x[:, :3], x[:, 3:6]
        if features is None:
            if outer is None:
                outer, valid = (i0, i1), (0., 0.)
            features = [self.context.encode(im) for im in (outer[0], i0, i1, outer[1])]
        # Match every base operation and dtype until the final warp. Frozen stages
        # do not retain activations, while context/final warps remain differentiable.
        with torch.no_grad():
            f0, f1 = self.base.encode(i0), self.base.encode(i1)
            t = (x[:, :1].clone() * 0 + 1) * .5
            flows, merged = [], []
            a, b, flow, mask = i0, i1, None, None
            for stage, block in enumerate((self.base.block0, self.base.block1,
                                            self.base.block2, self.base.block3)):
                if flow is None:
                    flow, mask = block(torch.cat((i0, i1, f0, f1, t), 1), None, scale=scales[stage])
                else:
                    wf0, wf1 = self.warp(f0, flow[:, :2]), self.warp(f1, flow[:, 2:])
                    df, mask = block(torch.cat((a, b, wf0, wf1, t, mask), 1), flow, scale=scales[stage])
                    flow = flow + df
                flows.append(flow)
                if stage < 3:
                    a, b = self.warp(i0, flow[:, :2]), self.warp(i1, flow[:, 2:])
                    merged.append((a, b))
        df, dm, residual = self.context(features, flow, mask, valid, self.warp)
        flow = flow + df.to(flow.dtype)
        mask = mask + dm.to(mask.dtype)
        a, b = self.warp(i0, flow[:, :2]), self.warp(i1, flow[:, 2:])
        m = mask.sigmoid()
        pred = (a * m + b * (1 - m) + residual).clamp(0, 1)
        return [*flows[:-1], flow], mask, [*merged, pred]


def selftest():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))
        def forward(self, x, flow=None, scale=1):
            f = x.new_zeros((x.shape[0], 4, *x.shape[-2:])) + self.bias
            return f, x.new_zeros((x.shape[0], 1, *x.shape[-2:]))
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode = nn.Conv2d(3, 8, 1)
            self.block0, self.block1, self.block2, self.block3 = [Block() for _ in range(4)]
    torch.manual_seed(7)
    torch.set_num_threads(2)
    net = ContextRIFE(Base())
    x, p, n = torch.rand(2, 6, 64, 96), torch.rand(2, 3, 64, 96), torch.rand(2, 3, 64, 96)
    x = F.interpolate(F.avg_pool2d(x, 8), (64, 96), mode="bilinear", align_corners=False)
    out = net(x, outer=(p, n))[2][-1]
    expected = (warp(x[:, :3], torch.zeros(2, 2, 64, 96)) + warp(x[:, 3:], torch.zeros(2, 2, 64, 96))) * .5
    assert torch.equal(out, expected), (out - expected).abs().max()
    target = (expected + .02).clamp(0, 1)
    opt = torch.optim.Adam(net.context.parameters(), lr=.0001)
    before = (out - target).square().mean()
    before.backward()
    assert all(t.grad is None for t in net.base.parameters())
    assert net.context.out.weight.grad.abs().sum() > 0
    opt.step()
    after = (net(x, outer=(p, n))[2][-1] - target).square().mean()
    assert after < before, (before, after)
    opt.zero_grad()
    after.backward()
    assert net.context.encoder[0].weight.grad.abs().sum() > 0
    assert net.context.align[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        explicit = net(x, outer=(x[:, :3], x[:, 3:]), valid=(0., 0.))[2][-1]
        assert torch.equal(net(x)[2][-1], explicit)
        changed = net(x, outer=(p.flip(0), n.flip(0)))[2][-1]
        normal = net(x, outer=(p, n))[2][-1]
        assert not torch.equal(changed, normal), "trained head must depend on outer inputs"
        features = [net.context.encode(z) for z in (p, x[:, :3], x[:, 3:], n)]
        assert torch.equal(net(x, features=features)[2][-1], normal)
    print(f"PASS identity/frozen-base/learning/context-gradients/edges/cache/context-dependence; "
          f"MSE {float(before.detach()):.8f}->{float(after.detach()):.8f}; "
          f"parameters={sum(t.numel() for t in net.context.parameters())}")


if __name__ == "__main__":
    selftest()
