"""Fast MVSFormer++ inference.

Same weights / state_dict as DINOv2MVSNet (subclass), same eval path (bf16 autocast, fp32 cost volumes).
  friendly: plain PyTorch only (ONNX-exportable ops): BN folded, all views batched through encoder / decoder / FMT /
            CrossVIT decoder, channels-last convs, no checkpoint wrappers, bf16 FPN upsampling, per-source cost volume
            with fused-friendly op order.
  max:      friendly + fused CUDA cost-volume kernels (warp + group correlation + softmax entropy, then weighted
            aggregation, fp32 math, no [C,D,H,W] intermediates) + vendored fastbb DINOv2 blocks + small-channel encoder conv
            kernel + fused last FPN level (exact linear rewrite) + head-dim-16 flash attention (stage-1 cost transformer).
Batch size 1 (the timed unit is one reference view + its sources).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module import (Conv2d, Conv3d, Deconv3d, ConvBnReLU, CostRegNet, CostRegNet3D, depth_regression,
                           init_inverse_range, schedule_inverse_range)
from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet
from models.position_encoding import get_position_3d

CL = torch.channels_last
CL3 = torch.channels_last_3d


@torch.no_grad()
def _fold(conv, bn, transposed=False):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight
    shape = ([1, -1] if transposed else [-1, 1]) + [1] * (w.dim() - 2)
    conv.weight = nn.Parameter(w * s.view(shape), requires_grad=False)
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    conv.bias = nn.Parameter((b - bn.running_mean) * s + bn.bias, requires_grad=False)


_CONVS = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_BNS = (nn.BatchNorm2d, nn.BatchNorm3d)


def fold_all(model):
    """Fold every eval BatchNorm into its preceding conv (upstream wrappers + nn.Sequential(conv, bn, act))."""
    n = 0
    for m in model.modules():
        if isinstance(m, (Conv2d, Conv3d, Deconv3d)) and getattr(m, "bn", None) is not None and isinstance(m.bn, _BNS):
            _fold(m.conv, m.bn, isinstance(m.conv, (nn.ConvTranspose2d, nn.ConvTranspose3d)))
            m.bn = None
            n += 1
        elif isinstance(m, ConvBnReLU) and isinstance(m.bn, _BNS):
            _fold(m.conv, m.bn)
            m.bn = nn.Identity()
            n += 1
        elif isinstance(m, nn.Sequential):
            for i in range(len(m) - 1):
                if isinstance(m[i], _CONVS) and isinstance(m[i + 1], _BNS):
                    _fold(m[i], m[i + 1], isinstance(m[i], (nn.ConvTranspose2d, nn.ConvTranspose3d)))
                    m[i + 1] = nn.Identity()
                    n += 1
    return n


class FastMVSFormerPP(DINOv2MVSNet):
    def __init__(self, args, mode="friendly"):
        super().__init__(args)
        self.mode = mode
        self.ext = None

    @torch.no_grad()
    def prepare(self):
        self.eval()
        self.n_folded = fold_all(self)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.weight.dim() == 4:
                m.weight.data = m.weight.data.contiguous(memory_format=CL)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=CL3)
        if self.mode.startswith("max"):
            from models import mvsf_ext
            self.ext = mvsf_ext.load()
            self._prep_fpn_max()
            if "novit" not in self.mode:  # fp16 DINOv2 blocks, flash64 attention, fp16-accumulate GEMMs
                from models.fastbb_vendored.vit import install_vit_blocks
                self.n_fastbb = install_vit_blocks(self.vit, "hh-g1" if "hh" in self.mode else "fh-g1")
            self.n_flash16 = self._install_flash16()
        return self

    def _install_flash16(self):
        """Head-dim-16 self-attention of the stage-1 cost transformer -> flash16 kernel (fp16 in, fp32 softmax/PV)."""
        from models.dino.layers.attention import FlashAttention2
        ext, n = self.ext, 0
        for m in self.modules():
            if type(m) is not FlashAttention2 or m.qkv.out_features // 3 // m.num_heads != 16:
                continue

            def fwd(x, attn_bias=None, positions=None, return_attn=False, m=m):
                B, N, C = x.shape
                qkv = m.qkv(x).to(torch.half).reshape(B, N, 3, m.num_heads, 16)
                sc = m.scale if m.softmax_scale is None else m.scale * math.log(N, m.train_avg_length)
                return m.proj(ext.flash16(qkv, sc, False).to(x.dtype if x.dtype != torch.float else torch.bfloat16))

            m.forward = fwd
            n += 1
        return n

    @torch.no_grad()
    def _prep_fpn_max(self):
        """Small-channel encoder convs -> conv_small kernel; last decoder level -> fpn_last (exact linear rewrite)."""
        self.enc_small = {}
        for name in ("conv00", "conv01", "downsample1", "conv10", "conv11"):
            m = getattr(self.encoder, name)
            w = m.conv.weight.detach().float()
            key = (w.shape[1], w.shape[0], w.shape[2], m.conv.stride[0])
            if key in {(3, 8, 7, 1), (8, 8, 5, 1), (8, 16, 5, 2), (16, 16, 3, 1)} and m.relu and m.bn is None:
                self.enc_small[name] = (w.permute(2, 3, 1, 0).contiguous(), m.conv.bias.detach().float().contiguous(),
                                        m.conv.stride[0], m.conv.padding[0])
        dec = self.decoder
        W3 = dec.out3[0].weight.detach().double()  # [CO, 64, 3, 3] (BN folded)
        b3 = dec.out3[0].bias.detach().double()
        Wi = dec.inner3.weight.detach().double()[:, :, 0, 0]  # [64, CI]
        bi = dec.inner3.bias.detach().double()
        CO = W3.shape[0]
        W3k = W3.permute(2, 3, 0, 1).reshape(9, CO, -1)  # [k, CO, 64]
        self.fpn_Wu = W3k.reshape(9 * CO, -1, 1, 1).to(torch.bfloat16).contiguous(memory_format=CL)
        Wc = torch.einsum("koc,ci->kio", W3k, Wi)  # [9, CI, CO]
        bk = torch.einsum("koc,c->ko", W3k, bi)  # [9, CO]
        self.fpn_prm = torch.cat([Wc.reshape(-1), bk.reshape(-1), b3]).float().contiguous()
        self.fpn_last_ok = isinstance(dec.out3[2], nn.Module) and type(dec.out3[2]).__name__ == "Swish" \
            and isinstance(dec.out3[1], nn.Identity) and CO == 8 and Wi.shape[1] == 8

    def _enc(self, x):
        e = self.encoder
        small = getattr(self, "enc_small", {}) if self.ext is not None else {}

        def c(name, t):
            if name in small:
                w, b, s, p = small[name]
                return self.ext.conv_small(t.contiguous(memory_format=CL), w, b, s, p)
            return getattr(e, name)(t)
        c01 = c("conv01", c("conv00", x))
        c11 = c("conv11", c("conv10", c("downsample1", c01)))
        c21 = e.conv21(e.conv20(e.downsample2(c11)))
        c31 = e.conv31(e.conv30(e.downsample3(c21)))
        return c01, c11, c21, c31

    # ------------------------------------------------------------------ features
    def _vit(self, x, V):
        H, W = x.shape[-2:]
        r = self.vit_args["rescale"]
        vh, vw = int(H * r // 14 * 14), int(W * r // 14 * 14)
        vimg = F.interpolate(x, (vh, vw), mode="bicubic", align_corners=False)
        out = self.vit.forward_interval_features(vimg)  # list of [V, HW, C]
        d = self.decoder_vit
        ref = [out[0][0:1]]
        for i in range(1, len(d.self_attn_blocks) + 1):
            new = d.prev_values[i - 1] * d.self_attn_blocks[i - 1](x=ref[-1]) + out[i][0:1]
            ref.append(d.norm_layers[i - 1](new) if not d.no_combine_norm else new)
        src = None
        S = V - 1
        for i in range(len(d.cross_attn_blocks)):
            if i == 0:
                q = out[i][1:]
            else:
                q = d.prev_values[i - 1] * src + out[i][1:]
                if not d.no_combine_norm:
                    q = d.norm_layers[i - 1](q)
            kv = ref[i].expand(S, -1, -1)
            src = d.cross_attn_blocks[i](x=q, key=kv, value=kv)
        hp, wp = vh // 14, vw // 14
        f = torch.cat([ref[-1], src], 0).reshape(V, hp, wp, -1).permute(0, 3, 1, 2)
        f = d.upsampler1(d.upsampler0(d.proj(f.contiguous(memory_format=CL))))
        if f.shape[2] != H // 8 or f.shape[3] != W // 8:
            f = F.interpolate(f, size=(H // 8, W // 8), mode="bilinear", align_corners=False)
        return f

    def _fpn(self, x, vit_feat):
        c01, c11, c21, c31 = self._enc(x)
        c31 = c31 + vit_feat
        dec = self.decoder
        out0 = dec.out0(c31)
        # upstream upsamples in fp32 (the next conv casts to bf16 anyway); bf16 here halves the traffic
        t = F.interpolate(c31, scale_factor=2, mode="bilinear", align_corners=True) + dec.inner1(c21)
        out1 = dec.out1(t)
        t = F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=True) + dec.inner2(c11)
        out2 = dec.out2(t)
        if self.ext is not None and getattr(self, "fpn_last_ok", False):
            U = F.conv2d(t.to(torch.bfloat16), self.fpn_Wu).contiguous(memory_format=CL)  # [V, 9*CO, h, w]
            out3 = self.ext.fpn_last(U, c01.to(torch.bfloat16).contiguous(memory_format=CL), self.fpn_prm)
        else:  # per view: the [64, H, W] full-res intermediates of all views at once cost ~5 GB
            out3 = torch.cat([dec.out3(F.interpolate(t[v:v + 1], scale_factor=2, mode="bilinear", align_corners=True)
                                       + dec.inner3(c01[v:v + 1])) for v in range(t.shape[0])], 0)
        return [out0, out1, out2, out3]

    def _fmt(self, f):
        fm = self.FMT_module
        fmt = fm.FMT
        S = f[0].shape[0] - 1
        ref_list = fmt.forward(f[0][0:1], feat="ref")
        src = fmt.forward([r.expand(S, -1, -1, -1) for r in ref_list], f[0][1:], feat="src")
        s1 = torch.cat([ref_list[-1], src], 0)
        s2 = fm.smooth_1(fm._upsample_add(fm.dim_reduction_1(s1), f[1]))
        s3 = fm.smooth_2(fm._upsample_add(fm.dim_reduction_2(s2), f[2]))
        s4 = fm.smooth_3(fm._upsample_add(fm.dim_reduction_3(s3), f[3]))
        return [s1, s2, s3, s4]

    # ------------------------------------------------------------------ cost volume
    @staticmethod
    def _rel_proj(P):
        """P [V,2,4,4] (w2c, K) -> src-from-ref [S,4,4] as homo_warping_3D_with_mask (fp32)."""
        M = P[:, 0].clone()
        M[:, :3, :4] = torch.matmul(P[:, 1, :3, :3], P[:, 0, :3, :4])
        return torch.matmul(M[1:], torch.inverse(M[0:1]))

    def _volume_friendly(self, st, f, proj, dv):
        """fp32 cost volume as StageNet (group correlation, entropy visibility, weighted mean)."""
        V, C, H, W = f.shape
        S = V - 1
        G = st.args["base_ch"][st.stage_idx] if isinstance(st.args["base_ch"], list) else st.args["base_ch"]
        D = dv.shape[1]
        rel = self._rel_proj(proj)
        ys, xs = torch.meshgrid(torch.arange(H, device=f.device, dtype=torch.float32),
                                torch.arange(W, device=f.device, dtype=torch.float32), indexing="ij")
        xyz = torch.stack((xs.reshape(-1), ys.reshape(-1), torch.ones(H * W, device=f.device)))  # [3, HW]
        rxyz = torch.matmul(rel[:, :3, :3], xyz)  # [S,3,HW]
        ref = f[0:1].float()
        vsum, wsum = None, None
        for s in range(S):
            p = rxyz[s].unsqueeze(1) * dv.view(1, D, H * W) + rel[s, :3, 3:4].view(3, 1, 1)  # [3,D,HW]
            xy = p[:2] / (p[2:3] + 1e-6)
            gx = xy[0] / ((W - 1) / 2) - 1
            gy = xy[1] / ((H - 1) / 2) - 1
            grid = torch.stack((gx, gy), 2).view(1, D * H, W, 2)
            warped = F.grid_sample(f[s + 1:s + 2].float(), grid, mode="bilinear", padding_mode="zeros",
                                   align_corners=True).view(1, C, D, H, W)
            if G < C:
                k = C // G
                wv, rv = warped.view(G, k, D, H, W), ref.view(G, k, 1, H, W)
                ip = wv[:, 0] * rv[:, 0]
                for j in range(1, k):
                    ip.addcmul_(wv[:, j], rv[:, j])
                ip = ip.div_(k).unsqueeze(0)  # [1,G,D,H,W]
            else:
                ip = warped.mul_(ref.view(1, G, 1, H, W))
            sim = ip.sum(1)
            pn = F.softmax(sim, dim=1)
            ent = (-pn * torch.log(pn + 1e-7)).sum(dim=1, keepdim=True)
            vis = st.vis(ent).unsqueeze(1)  # [1,1,1,H,W]
            if vsum is None:
                vsum, wsum = ip.mul_(vis), vis.clone()
            else:
                vsum.addcmul_(ip, vis)
                wsum += vis
        return vsum / (wsum + 1e-6)

    def _volume_max(self, st, f, proj, dv):
        V, C, H, W = f.shape
        G = st.args["base_ch"][st.stage_idx] if isinstance(st.args["base_ch"], list) else st.args["base_ch"]
        rel = self._rel_proj(proj)[:, :3, :].reshape(V - 1, 12).contiguous()
        fcl = f.contiguous(memory_format=CL)
        dvc = dv.reshape(-1, H * W).float().contiguous()
        ent = self.ext.entropy(fcl, rel, dvc, G)  # [S,1,H,W] fp32
        vis = st.vis(ent)  # [S,1,H,W]
        return self.ext.aggregate(fcl, rel, dvc, vis.reshape(V - 1, H * W).contiguous(), G)  # [1,G,D,H,W] fp32

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, imgs, proj_matrices, depth_values, tmp=(5.0, 5.0, 5.0, 1.0)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._forward(imgs, proj_matrices, depth_values, tmp)

    def _forward(self, imgs, proj_matrices, depth_values, tmp):
        B, V, _, H, W = imgs.shape
        assert B == 1, "batch 1"
        x = imgs[0].contiguous(memory_format=CL)
        feats = self._fmt(self._fpn(x, self._vit(imgs[0], V)))
        prob_maps = torch.zeros([1, H, W], dtype=torch.float32, device=imgs.device)
        hmin = hmax = wmin = wmax = None
        out = None
        for si in range(len(self.ndepths)):
            P = proj_matrices[f"stage{si + 1}"][0]
            f = feats[si]
            h, w = f.shape[2:]
            if si == 0:
                dv = init_inverse_range(depth_values, self.ndepths[si], imgs.device, imgs.dtype, h, w)
            else:
                dv = schedule_inverse_range(out[0], out[1], self.ndepths[si], self.depth_interals_ratio[si], h, w)
            st = self.fusions[si]
            pos = None
            if self.cost_reg_type[si] != "Normal" and self.use_pe3d:
                pos, hmin, hmax, wmin, wmax = get_position_3d(
                    1, h, w, P[0:1, 1, :3, :3], dv, depth_min=depth_values.min(), depth_max=depth_values.max(),
                    height_min=hmin, height_max=hmax, width_min=wmin, width_max=wmax, normalize=True)
            with torch.autocast("cuda", enabled=False):
                vol = self._volume_max(st, f, P, dv) if self.ext is not None else self._volume_friendly(st, f, P, dv)
            reg = st.cost_reg
            if isinstance(reg, (CostRegNet, CostRegNet3D)):
                pre = reg.forward_once(vol.contiguous(memory_format=CL3))
            else:
                pre = reg(vol, pos)
            pre = pre.squeeze(1)
            prob = F.softmax(pre, dim=1)
            depth = depth_regression(F.softmax(pre * tmp[si], dim=1), depth_values=dv)
            conf = prob.max(1)[0]
            if conf.shape[1] != H or conf.shape[2] != W:
                conf = F.interpolate(conf.unsqueeze(1), [H, W], mode="nearest").squeeze(1)
            prob_maps += conf
            out = (depth, dv)
        return out[0], prob_maps / len(self.ndepths)
