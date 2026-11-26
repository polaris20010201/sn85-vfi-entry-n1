"""Four-input context with cached neighboring RIFE flows. Original code, MIT."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
import torch.nn.functional as F
from context import ContextHead, ContextRIFE


class MotionHead(ContextHead):
    def __init__(self, **config):
        super().__init__(**config)
        del self.align
        cin=4*(self.channels+3)+4+1+2+6
        self.e0[0]=nn.Conv2d(cin,self.widths[0],3,padding=1)
        self.out=nn.Conv2d(self.widths[0],9,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def geometry(self, flow, neighbors, valid, size, warp_fn):
        fq=F.interpolate(flow,size,mode='bilinear',align_corners=False)*.25
        vs=[]
        for v in valid:
            if not torch.is_tensor(v):v=fq.new_full((fq.shape[0],1,1,1),float(v))
            vs.append(v.to(fq).expand(-1,1,*size))
        vp,vn=vs
        pq,nq=neighbors
        if pq is None:pq=torch.zeros_like(fq)
        if nq is None:nq=torch.zeros_like(fq)
        dl=warp_fn((pq[:,:2]-pq[:,2:]).float(),fq[:,:2]-pq[:,2:])*vp
        dr=warp_fn((nq[:,2:]-nq[:,:2]).float(),fq[:,2:]-nq[:,:2])*vn
        motion=fq[:,2:]-fq[:,:2]
        curvature=((motion+dl)*vp+(dr-motion)*vn)/(8*(vp+vn).clamp_min(1))
        return fq,dl,dr,curvature,vp,vn

    def fuse(self, features, flow, logit, valid, warp_fn, neighbors):
        size=features[0].shape[-2:]
        fq,dl,dr,curvature,vp,vn=self.geometry(flow,neighbors,valid,size,warp_fn)
        fs=[fq[:,:2]+dl,fq[:,:2],fq[:,2:],fq[:,2:]+dr]
        aligned=[warp_fn(f.float(),d) for f,d in zip(features,fs)]
        mq=F.interpolate(logit,size,mode='bilinear',align_corners=False).sigmoid()
        a=self.e0(torch.cat((*aligned,fq/8,mq,vp,vn,dl/8,dr/8,curvature/2),1))
        b=self.e1(a)
        c=self.e2(b)
        b=self.d1(torch.cat((b,F.interpolate(c,b.shape[-2:],mode='bilinear',align_corners=False)),1))
        a=self.d0(torch.cat((a,F.interpolate(b,a.shape[-2:],mode='bilinear',align_corners=False)),1))
        return self.out(a),curvature,dl,dr,vp,vn

    def forward(self, features, flow, logit, valid, warp_fn, neighbors=(None,None)):
        raw,curvature,dl,dr,vp,vn=self.fuse(features,flow,logit,valid,warp_fn,neighbors)
        delta=F.interpolate(raw,flow.shape[-2:],mode='bilinear',align_corners=False).float()
        prior=(4*F.interpolate(curvature,flow.shape[-2:],mode='bilinear',align_corners=False)).clamp(-self.flow_limit,self.flow_limit)
        prior=torch.cat((prior,prior),1)/self.flow_limit
        df=self.flow_limit*(delta[:,:4]+prior*delta[:,8:9].tanh()).tanh()
        basic=(df,2*delta[:,4:5].tanh(),self.residual_limit*delta[:,5:8].tanh())
        if self.out.out_channels>9:
            validity=F.interpolate(torch.cat((vp,vn),1),flow.shape[-2:],mode='nearest')
            taps=self.outer_limit*delta[:,9:11].tanh()*validity
            transport=4*F.interpolate(torch.cat((dl,dr),1),flow.shape[-2:],mode='bilinear',align_corners=False)
            return (*basic,taps,transport)
        return basic


class MotionRIFE(ContextRIFE):
    def __init__(self, base, **config):
        super().__init__(base,**config)
        self.context=MotionHead(**config)
        self.primary_calls=0

    def quarter_flow(self, flow, size=None):
        if size is None:size=(flow.shape[-2]//4,flow.shape[-1]//4)
        return F.interpolate(flow,size,mode='bilinear',align_corners=False)*.25

    @torch.no_grad()
    def primary(self, x, scales=None):
        self.primary_calls+=1
        scales=[8.,4.,2.,1.] if scales is None else scales
        i0,i1=x[:,:3],x[:,3:6]
        f0,f1=self.base.encode(i0),self.base.encode(i1)
        t=(x[:,:1].clone()*0+1)*.5
        a,b,flow,mask=i0,i1,None,None
        for stage,block in enumerate((self.base.block0,self.base.block1,self.base.block2,self.base.block3)):
            if flow is None:flow,mask=block(torch.cat((i0,i1,f0,f1,t),1),None,scale=scales[stage])
            else:
                wf0,wf1=self.warp(f0,flow[:,:2]),self.warp(f1,flow[:,2:])
                df,mask=block(torch.cat((a,b,wf0,wf1,t,mask),1),flow,scale=scales[stage])
                flow=flow+df
            if stage<3:a,b=self.warp(i0,flow[:,:2]),self.warp(i1,flow[:,2:])
        return flow,mask

    def synthesize(self, x, fields, features, valid=(1.,1.), neighbors=(None,None), outer=None):
        flow,mask=fields
        df,dm,residual=self.context(features,flow,mask,valid,self.warp,neighbors)
        flow,mask=flow+df.to(flow.dtype),mask+dm.to(mask.dtype)
        m=mask.sigmoid()
        pred=(self.warp(x[:,:3],flow[:,:2])*m+self.warp(x[:,3:6],flow[:,2:])*(1-m)+residual).clamp(0,1)
        return [flow],mask,[pred]

    def forward(self, x, timestep=.5, scale_list=None, *, outer=None, features=None,
                valid=(1.,1.), fields=None, neighbors=None, **kwargs):
        if timestep!=.5 or kwargs.get('ensemble',False):raise ValueError('midpoint only')
        i0,i1=x[:,:3],x[:,3:6]
        no_outer=outer is None
        if no_outer:outer,valid=(i0,i1),(0.,0.)
        if features is None:features=[self.context.encode(im) for im in (outer[0],i0,i1,outer[1])]
        if fields is None:fields=self.primary(x,scale_list)
        if neighbors is None and no_outer:neighbors=(None,None)
        if neighbors is None:
            prev=self.primary(torch.cat((outer[0],i0),1),scale_list)[0]
            nxt=self.primary(torch.cat((i1,outer[1]),1),scale_list)[0]
            neighbors=(self.quarter_flow(prev),self.quarter_flow(nxt))
        return self.synthesize(x,fields,features,valid,neighbors,outer)


class FourWarpHead(MotionHead):
    def __init__(self,outer_limit=.5,hires_channels=0,**config):
        super().__init__(**config)
        if not isinstance(hires_channels,int) or hires_channels<0:
            raise ValueError('hires_channels must be a nonnegative integer')
        self.hires_channels=hires_channels
        self.outer_limit=outer_limit
        self.out=nn.Conv2d(self.widths[0],11,1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        if hires_channels:
            from context import Sep
            self.hires_encoder=nn.Sequential(nn.Conv2d(3,hires_channels,3,padding=1),
                                            nn.LeakyReLU(.2),Sep(hires_channels,hires_channels))
            self.hires_fuse=nn.Sequential(
                nn.Conv2d(4*(hires_channels+3)+13,hires_channels,3,padding=1),
                nn.LeakyReLU(.2),Sep(hires_channels,hires_channels))
            self.hires_out=nn.Conv2d(hires_channels,11,1)
            nn.init.zeros_(self.hires_out.weight)
            nn.init.zeros_(self.hires_out.bias)

    def config(self):
        return dict(super().config(),outer_limit=self.outer_limit,
                    hires_channels=self.hires_channels)

    def encode(self,rgb):
        low=super().encode(rgb)
        if not self.hires_channels:return low
        half=F.avg_pool2d(rgb,2)
        return low,torch.cat((self.hires_encoder(half),half),1)

    def forward(self,features,flow,logit,valid,warp_fn,neighbors=(None,None)):
        low=[f[0] for f in features] if self.hires_channels else features
        raw,curvature,dl,dr,vp,vn=self.fuse(low,flow,logit,valid,warp_fn,neighbors)
        if self.hires_channels:
            high=[f[1] for f in features]
            size=high[0].shape[-2:]
            def resize(t):
                return F.interpolate(t,size,mode='bilinear',align_corners=False)
            # Geometry is expressed in quarter-resolution pixels; convert to half.
            fq=F.interpolate(flow,size,mode='bilinear',align_corners=False)*.5
            hd,hn=2*resize(dl),2*resize(dr)
            offsets=(fq[:,:2]+hd,fq[:,:2],fq[:,2:],fq[:,2:]+hn)
            aligned=[warp_fn(f.float(),d) for f,d in zip(high,offsets)]
            guide=(fq/16,F.interpolate(logit,size,mode='bilinear',align_corners=False).sigmoid(),
                   resize(vp),resize(vn),hd/16,hn/16,resize(curvature)/2)
            correction=self.hires_out(self.hires_fuse(torch.cat((*aligned,*guide),1)))
            raw=resize(raw)+correction
            curvature,dl,dr,vp,vn=[resize(t) for t in (curvature,dl,dr,vp,vn)]
        raw=raw.float()
        prior=(4*curvature).clamp(-self.flow_limit,self.flow_limit)
        prior=torch.cat((prior,prior),1)/self.flow_limit
        df=self.flow_limit*(raw[:,:4]+prior*raw[:,8:9].tanh()).tanh()
        dm,res=2*raw[:,4:5].tanh(),self.residual_limit*raw[:,5:8].tanh()
        taps=self.outer_limit*raw[:,9:11].tanh()*torch.cat((vp,vn),1)
        packed=torch.cat((df,dm,res,taps,4*dl,4*dr),1)
        full=F.interpolate(packed,flow.shape[-2:],mode='bilinear',align_corners=False)
        return full[:,:4],full[:,4:5],full[:,5:8],full[:,8:10],full[:,10:14]


class FourWarpRIFE(MotionRIFE):
    def __init__(self,base,outer_limit=.5,channels=8,widths=(16,24,32),hires_channels=0,**config):
        common=dict(channels=channels,widths=widths,**config)
        super().__init__(base,**common)
        self.context=FourWarpHead(outer_limit=outer_limit,hires_channels=hires_channels,**common)

    def synthesize(self,x,fields,features,valid=(1.,1.),neighbors=(None,None),outer=None):
        if outer is None:outer=(x[:,:3],x[:,3:6])
        flow,mask=fields
        df,dm,residual,taps,transport=self.context(features,flow,mask,valid,self.warp,neighbors)
        flow,mask=flow+df.to(flow.dtype),mask+dm.to(mask.dtype)
        m=mask.sigmoid()
        central=self.warp(x[:,:3],flow[:,:2])*m+self.warp(x[:,3:6],flow[:,2:])*(1-m)
        p=self.warp(outer[0],flow[:,:2]+transport[:,:2].to(flow.dtype))
        n=self.warp(outer[1],flow[:,2:]+transport[:,2:].to(flow.dtype))
        pred=(central+residual+taps[:,:1]*(p-central)+taps[:,1:]*(n-central)).clamp(0,1)
        return [flow],mask,[pred]


def selftest():
    from context import warp
    torch.manual_seed(5)
    torch.set_num_threads(2)
    h,w=64,96
    head=MotionHead()
    f=torch.tensor([-4.,-2.,4.,2.]).view(1,4,1,1).expand(1,4,h,w)
    p=torch.tensor([0.,-4.,0.,4.]).view(1,4,1,1).expand(1,4,h//4,w//4)/4
    n=torch.tensor([-8.,0.,8.,0.]).view(1,4,1,1).expand(1,4,h//4,w//4)/4
    expected=torch.tensor([1.,-.5]).view(1,2,1,1)
    for valid in [(1.,1.),(1.,0.),(0.,1.)]:
        geom=head.geometry(f,(p,n),valid,(h//4,w//4),warp)
        assert torch.allclose(4*geom[3],expected.expand_as(geom[3]),atol=1e-6)
    assert head.geometry(f,(p,n),(0.,0.),(h//4,w//4),warp)[3].abs().max()==0
    features=[head.encode(torch.rand(1,3,h,w)) for _ in range(4)]
    out=head(features,f,torch.zeros(1,1,h,w),(1.,1.),warp,(p,n))
    assert all(torch.count_nonzero(z)==0 for z in out)
    # A confidence bias alone must expose the known trajectory prior with its correct sign.
    with torch.no_grad():head.out.bias[8]=.5
    df,dm,res=head(features,f,torch.zeros(1,1,h,w),(1.,1.),warp,(p,n))
    assert df[:,0].mean()>0 and df[:,1].mean()<0
    assert torch.equal(df[:,:2],df[:,2:]) and dm.abs().max()==res.abs().max()==0
    print('MOTION CPU PASS: exact quadratic curvature (both/single sides), absent context, zero corrections, prior direction; parameters',sum(p.numel() for p in head.parameters()))


if __name__=='__main__':selftest()
