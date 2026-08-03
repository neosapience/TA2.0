# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from copy import deepcopy
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import PeftAdapterMixin
from .attention import flash_attention
from einops import repeat, rearrange

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000, scale=1.0):
    """
    scale < 1.0 : 속도 느리게 (예: 0.5 → 절반 속도)
    scale > 1.0 : 속도 빠르게
    """
    assert dim % 2 == 0
    # 1) t 인덱스에 스케일 적용
    t = torch.arange(max_seq_len, dtype=torch.float64) * scale
    inv_freq = 1.0 / torch.pow(theta,
                              torch.arange(0, dim, 2, dtype=torch.float64).div(dim))
    # 2) outer로 각도 계산
    angles = torch.outer(t, inv_freq)
    # 3) 복소수 unit vector 생성
    freqs = torch.polar(torch.ones_like(angles), angles)
    return freqs

@torch.amp.autocast("cuda", enabled=False)
def rope_apply_ref_frames(x, grid_sizes, freqs, ref_frame_num=0):
    if ref_frame_num == 0:
        return None

    # x: [B, seq_len, n, 2*c] -> extract n and c
    n, c = x.size(2), x.size(3) // 2  # n: number of heads, c: head dimension
    bs = x.size(0)  # batch size

    # split freqs: [seq_len, c] -> [seq_len, c-2*(c//3)], [seq_len, c//3], [seq_len, c//3]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # extract grid dimensions
    f, h, w = grid_sizes[0].tolist()  # frames, height, width
    seq_len = f * h * w  # total sequence length

    assert ref_frame_num*h*w == x.size(1), f"ref_frame_num*h*w: {ref_frame_num*h*w}, x.size(1): {x.size(1)}"

    # precompute multipliers
    # x: [B, seq_len, n, 2*c] -> [B, seq_len, n, c, 2] -> [B, seq_len, n, c] (complex format)
    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, ref_frame_num*h*w, n, -1, 2))

    # Create frequency tensors for each dimension and concatenate
    # freqs[0][-ref_frame_num:]: [ref_frame_num, c-2*(c//3)] -> [ref_frame_num, 1, 1, c-2*(c//3)] -> [ref_frame_num, h, w, c-2*(c//3)]
    # freqs[1][:h]: [h, c//3] -> [1, h, 1, c//3] -> [ref_frame_num, h, w, c//3]
    # freqs[2][:w]: [w, c//3] -> [1, 1, w, c//3] -> [ref_frame_num, h, w, c//3]
    freqs_i = torch.cat(
        [
            freqs[0][-ref_frame_num:].view(ref_frame_num, 1, 1, -1).expand(ref_frame_num, h, w, -1),  # [ref_frame_num, h, w, c-2*(c//3)]
            freqs[1][:h].view(1, h, 1, -1).expand(ref_frame_num, h, w, -1),  # [ref_frame_num, h, w, c//3]
            freqs[2][:w].view(1, 1, w, -1).expand(ref_frame_num, h, w, -1),  # [ref_frame_num, h, w, c//3]
        ],
        dim=-1,  # [ref_frame_num, h, w, c]
    ).reshape(ref_frame_num*h*w, 1, -1)  # [ref_frame_num*h*w, 1, c]

    # apply rotary embedding
    # x: [B, seq_len, n, c] * freqs_i: [seq_len, 1, c] -> [B, seq_len, n, c]
    # torch.view_as_real: [B, seq_len, n, c] -> [B, seq_len, n, c, 2]
    # flatten(3): [B, seq_len, n, c, 2] -> [B, seq_len, n, c*2]

    x = torch.view_as_real(x * freqs_i).flatten(3)

    return x

@torch.amp.autocast("cuda", enabled=False)
def rope_apply_with_frame_idx(x, grid_sizes, freqs, frame_idx=None):
    if frame_idx is None:
        return None

    # x: [B, s_end, n, 2*c] where s_end = 1*h*w for end frames
    n, c = x.size(2), x.size(3) // 2  # n: number of heads, c: head dimension
    bs = x.size(0)  # batch size
    s_end = x.size(1)  # sequence length for end frames (should be 1*h*w)

    # split freqs: [seq_len, c] -> [seq_len, c-2*(c//3)], [seq_len, c//3], [seq_len, c//3]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # extract grid dimensions
    f, h, w = grid_sizes[0].tolist()  # frames, height, width
    seq_len = f * h * w  # total sequence length

    # Verify that s_end matches expected size for end frames
    assert s_end == 1 * h * w, f"s_end: {s_end}, expected: {1 * h * w}"

    # precompute multipliers
    # x: [B, s_end, n, 2*c] -> [B, s_end, n, c, 2] -> [B, s_end, n, c] (complex format)
    x = torch.view_as_complex(x.to(torch.float32).reshape(bs, s_end, n, -1, 2))

    # Batch-wise parallel operation for freqs_i

    # Create freqs_i for all batch samples at once
    # freqs[0][frame_idx]: [B, c-2*(c//3)] -> [B, 1, 1, 1, c-2*(c//3)] -> [B, 1, h, w, c-2*(c//3)]
    freqs_0 = freqs[0][frame_idx].view(bs, 1, 1, 1, -1).expand(bs, 1, h, w, -1)
    # freqs[1][:h]: [h, c//3] -> [1, 1, h, 1, c//3] -> [B, 1, h, w, c//3]
    freqs_1 = freqs[1][:h].view(1, 1, h, 1, -1).expand(bs, 1, h, w, -1)
    # freqs[2][:w]: [w, c//3] -> [1, 1, 1, w, c//3] -> [B, 1, h, w, c//3]
    freqs_2 = freqs[2][:w].view(1, 1, 1, w, -1).expand(bs, 1, h, w, -1)

    # Concatenate along the last dimension
    freqs_i = torch.cat([freqs_0, freqs_1, freqs_2], dim=-1)  # [B, 1, h, w, c]
    freqs_i = freqs_i.reshape(bs, s_end, 1, -1)  # [B, s_end, 1, c]

    # Apply rotary embedding for all batch samples at once
    # x: [B, s_end, n, c] * freqs_i: [B, s_end, 1, c] -> [B, s_end, n, c]
    x = x * freqs_i  # [B, s_end, n, c]

    # Convert back to real and flatten
    # torch.view_as_real: [B, s_end, n, c] -> [B, s_end, n, c, 2] (real format)
    # flatten(3): [B, s_end, n, c, 2] -> [B, s_end, n, c*2] (flattening last two dimensions)
    x = torch.view_as_real(x).flatten(3)  # [B, s_end, n, c*2]

    return x

@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs, temporal_offset=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))

        # temporal frequencies with optional offset
        if temporal_offset == -1:
            # position -1 = complex conjugate of position 1
            # positions: [-1, 0, 1, ..., f-2]  (Variant C / ref_end_mode="replace")
            temporal_freqs = torch.cat([freqs[0][1:2].conj(), freqs[0][:f-1]], dim=0)
        else:
            temporal_freqs = freqs[0][:f]

        freqs_i = torch.cat([
            temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, origin_seq_lens, grid_sizes, origin_grid_sizes, freqs, ref_frame_num=0, ref_frame_idx=None, end_frame_num=0, end_frame_idx=None, temporal_offset=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            origin_seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            origin_grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            temporal_offset(int): RoPE temporal position offset (0 default, -1 for ref_end_mode="replace")
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        origin_seq_len = torch.max(origin_seq_lens)
        ref_seq_len = torch.max(torch.prod(grid_sizes[:, 1:], dim=1) * ref_frame_num)
        end_seq_len = torch.max(torch.prod(grid_sizes[:, 1:], dim=1) * end_frame_num)

        # Use torch.split to separate q and k into origin, ref, and end segments
        split_sizes = [origin_seq_len.item(), ref_seq_len.item(), end_seq_len.item()]
        q_origin, q_ref, q_end = torch.split(q, split_sizes, dim=1)
        k_origin, k_ref, k_end = torch.split(k, split_sizes, dim=1)

        q_origin = rope_apply(q_origin, origin_grid_sizes, freqs, temporal_offset=temporal_offset)
        k_origin = rope_apply(k_origin, origin_grid_sizes, freqs, temporal_offset=temporal_offset)

        if ref_frame_num > 0:
            q_ref = rope_apply_with_frame_idx(q_ref, grid_sizes, freqs, frame_idx=ref_frame_idx).type_as(v)
            k_ref = rope_apply_with_frame_idx(k_ref, grid_sizes, freqs, frame_idx=ref_frame_idx).type_as(v)
        else:
            q_ref = None
            k_ref = None

        if end_frame_num > 0:
            q_end = rope_apply_with_frame_idx(q_end, grid_sizes, freqs, frame_idx=end_frame_idx).type_as(v)
            k_end = rope_apply_with_frame_idx(k_end, grid_sizes, freqs, frame_idx=end_frame_idx).type_as(v)
        else:
            q_end = None
            k_end = None


        q_list = [q_origin]
        k_list = [k_origin]
        if q_ref is not None:
            q_list.append(q_ref)
            k_list.append(k_ref)
        if q_end is not None:
            q_list.append(q_end)
            k_list.append(k_end)

        q = torch.cat(q_list, dim=1)
        k = torch.cat(k_list, dim=1)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens, ### Be careful with this
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttentionAudio(WanSelfAttention):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, audio_dim=768, hidden_dim=3072, audio_scale=0.0):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        self.audio_scale = audio_scale

    def forward(self, x, audio = None, audio_attn_output_mask = None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            audio(Tensor): Shape [B, Lv, La, C], audio embeddings
        """

        if self.audio_scale == 0 or audio is None:
            return 0

        b, n, d = x.size(0), self.num_heads, self.head_dim
        _, Lv, La, _ = audio.shape # [B, Lv, La, C]

        B = b * Lv

        x = rearrange(x, 'b (Lv Lhw) c -> (b Lv) Lhw c', Lv=Lv)
        audio = rearrange(audio, 'b Lv La c -> (b Lv) La c')

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(B, -1, n, d) # [B, Lv, Lhw, C]

        k = self.norm_k(self.k(audio)).view(B, -1, n, d) # [B, Lv, Lhw, C]
        v = self.v(audio).view(B, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v)

        # output
        x = x.flatten(2)
        x = self.o(x)

        x = rearrange(x, '(b Lv) Lhw c -> b (Lv Lhw) c', Lv=Lv)
        if audio_attn_output_mask is not None:
            x = x * audio_attn_output_mask

        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def init_audio_cross_attn_multi_talk(self, audio_dim=768, hidden_dim=3072, audio_scale=0.0):
        self.cross_attn_audio = WanCrossAttentionAudio(self.dim, self.num_heads, (-1, -1), self.qk_norm, self.eps, audio_dim=audio_dim, hidden_dim=hidden_dim, audio_scale=audio_scale)
        self.norm_audio = WanLayerNorm(self.dim, self.eps, elementwise_affine=True)

    def forward(
        self,
        x,
        e,
        seq_lens,
        origin_seq_lens,
        grid_sizes,
        origin_grid_sizes,
        freqs,
        context,
        context_lens,
        ref_frame_num,
        ref_frame_idx,
        end_frame_num,
        end_frame_idx,
        audio_embeds=None,
        audio_attn_output_mask=None,
        temporal_offset=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            origin_seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            origin_grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            ref_frame_num(int): Number of reference frames (e.g. 1)
            ref_frame_idx(Tensor): Shape [B]
            end_frame_num(int): Number of end frames (e.g. 1)
            end_frame_idx(Tensor): Shape [B]
            audio_embeds(Tensor, optional): Shape [B, Lv, La, D] Audio embeddings
            audio_attn_output_mask(Tensor, optional): Shape [B, origin_seq_len, 1] Audio attention mask
            temporal_offset(int): RoPE temporal position offset (0 default, -1 for ref_end_mode="replace")
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, origin_seq_lens, grid_sizes, origin_grid_sizes, freqs, ref_frame_num=ref_frame_num, ref_frame_idx=ref_frame_idx, end_frame_num=end_frame_num, end_frame_idx=end_frame_idx, temporal_offset=temporal_offset)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, audio_embeds, audio_attn_output_mask, origin_seq_lens):
            # Step 1: Text cross-attention on ENTIRE sequence
            x = x + self.cross_attn(self.norm3(x), context, context_lens)

            # Step 2: Audio cross-attention ONLY on origin sequence (if audio present)
            if hasattr(self, 'cross_attn_audio') and self.cross_attn_audio is not None and audio_embeds is not None:
                # Get origin boundary from stored tensor (NOT hardcoded!)
                origin_seq_len = origin_seq_lens.max().item()

                # Split at origin boundary
                x_origin = x[:, :origin_seq_len, :]  # Before boundary = origin
                x_rest = x[:, origin_seq_len:, :]     # After boundary = ref+end

                # Audio attention ONLY on origin
                x_origin = x_origin + self.cross_attn_audio(
                    self.norm_audio(x_origin),
                    audio_embeds,
                    audio_attn_output_mask
                )

                # Concatenate back in same order
                x = torch.cat([x_origin, x_rest], dim=1)

            # Step 3: FFN on entire sequence
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, audio_embeds, audio_attn_output_mask, origin_seq_lens)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    _supports_gradient_checkpointing = True


    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 time_scale=1.0,
                 ref_rope_idx=-1,
                 ref_end_mode="partial",
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.time_scale = time_scale
        self.ref_rope_idx = ref_rope_idx
        self.ref_end_mode = ref_end_mode

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), scale=self.time_scale),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        self.gradient_checkpointing = False


        self.concat_ref = False
        self.cond_patch_embedding = None

        # initialize weights
        self.init_weights()

    def init_cond_patch_embedding(self):
        self.cond_patch_embedding = deepcopy(self.patch_embedding)

    def zero_init_patch_embedding(self, new_in_dim):

        origin_weight = deepcopy(self.patch_embedding.weight.data)
        origin_bias = deepcopy(self.patch_embedding.bias.data)

        # Create new Conv3d layer with new input dimension
        self.patch_embedding = nn.Conv3d(new_in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Extend the original weight with zeros for the new input channels
        if origin_weight is not None:
            old_in_dim = origin_weight.shape[1]
            if new_in_dim > old_in_dim:
                # Create zero padding for additional input channels
                padding = torch.zeros(origin_weight.shape[0], new_in_dim - old_in_dim, *origin_weight.shape[2:],
                                    device=origin_weight.device, dtype=origin_weight.dtype)
                extended_weight = torch.cat([origin_weight, padding], dim=1)
                self.patch_embedding.weight.data = extended_weight
            else:
                self.patch_embedding.weight.data = origin_weight

        # Set bias (should remain the same since output dimension hasn't changed)
        if origin_bias is not None:
            self.patch_embedding.bias.data = origin_bias

        self.in_dim = new_in_dim
        self.concat_ref = True


    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        ref_latents=None,
        end_latents=None,
        end_frame_idx=None,
        audio_embeds=None,
        audio_attn_mask=None,
        audio_attn_output_mask=None,
        cond_pure_norm: bool = False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor] | Tensor):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor] | Tensor):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor] | Tensor, *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            ref_latents (Tensor, *optional*): [B, C, 1, H, W]
                Reference video inputs for image-to-video mode, same shape as x
            end_latents (Tensor, *optional*): [B, C, 1, H, W]
                End video inputs for image-to-video mode, same shape as x
            end_frame_idx (Tensor, *optional*): [B]
                End frame index for image-to-video mode, same shape as x
            audio_embeds (Tensor, *optional*): [B, T_a, C]
                Audio embeddings for audio-to-video mode
            audio_attn_mask (Tensor, *optional*): [B, T_a, T_v]
                Audio attention mask for audio-to-video mode
            audio_attn_output_mask (Tensor, *optional*): [B, T_v, T_a]
                Audio attention output mask for audio-to-video mode
            cond_pure_norm (bool, *optional*):
                Whether to use pure normalization for conditioning
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        is_input_tensor = isinstance(x, torch.Tensor)
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            raise NotImplementedError("y is not supported")
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if ref_latents is not None:
            # ref_latent: [B, C, T_ref, H, W]
            ref_frame_num = ref_latents.shape[2]
        else:
            ref_frame_num = 0

        if end_latents is not None:
            # end_latent: [B, C, 1, H, W]
            end_frame_num = end_latents.shape[2]
        else:
            end_frame_num = 0


        # embeddings
        if self.concat_ref:
            assert ref_latents is not None, "ref_latents must be provided when concat_ref is True"
            if not is_input_tensor:
                x = [torch.cat([u, ref_latents[batch_idx, :, -1:, :, :].repeat(1, u.shape[1], 1, 1)], dim=0) for batch_idx, u in enumerate(x)] # [2C, T, H, W]
            else:
                x = torch.cat([x, ref_latents[:, :, -1:, :, :].repeat(1, 1, x.shape[2], 1, 1)], dim=1) # [B, 2C, T, H, W]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] if not is_input_tensor else self.patch_embedding(x)
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]) if not is_input_tensor else torch.stack(
            [torch.tensor(u.shape[1:], dtype=torch.long) for u in x]) # [B, 3]
        x = [u.flatten(2).transpose(1, 2) for u in x] if not is_input_tensor else x.flatten(2).transpose(1, 2) # [B, seq_len, C]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long) if not is_input_tensor else torch.tensor([u.size(0) for u in x], dtype=torch.long)

        # assert seq_lens.max() <= seq_len
        # assert seq_lens.max() == seq_len and seq_lens.min() == seq_len, f"seq_lens: {seq_lens}, seq_len: {seq_len}"

        if not is_input_tensor:
            x = torch.cat(x, dim=0)


        if ref_latents is not None and self.cond_patch_embedding is not None:
            ref_latents = self.cond_patch_embedding(ref_latents) # [B, C, T_ref, H, W]
            ref_latents = ref_latents.flatten(2).transpose(1, 2) # [B, ref_seq_len, C]
            ref_seq_len = ref_latents.shape[1]
        else:
            ref_latents = None
            ref_seq_len = 0

        if end_latents is not None and self.cond_patch_embedding is not None:
            end_latents = self.cond_patch_embedding(end_latents) # [B, C, 1, H, W]
            end_latents = end_latents.flatten(2).transpose(1, 2) # [B, end_seq_len, C]
            end_seq_len = end_latents.shape[1]
        else:
            end_latents = None
            end_seq_len = 0


        origin_seq_lens = seq_lens.clone()
        origin_grid_sizes = grid_sizes.clone()

        if ref_latents is not None:
            x = torch.cat([x, ref_latents], dim=1)
            seq_lens = seq_lens + ref_seq_len
            grid_sizes[:, 0] = grid_sizes[:, 0] + 1

        if end_latents is not None:
            x = torch.cat([x, end_latents], dim=1)
            seq_lens = seq_lens + end_seq_len
            grid_sizes[:, 0] = grid_sizes[:, 0] + 1

        # time embeddings
        origin_seq_len = origin_seq_lens.max()
        seq_len = seq_lens.max()

        if t.dim() == 1:
            t = t.expand(t.size(0), origin_seq_len)

        bt = t.size(0)

        if ref_latents is not None:
            t_ref = torch.zeros((bt, ref_seq_len), dtype=t.dtype, device=t.device)
            t = torch.cat([t, t_ref], dim=1)

        if end_latents is not None:
            t_end = torch.zeros((bt, end_seq_len), dtype=t.dtype, device=t.device)
            t = torch.cat([t, t_end], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if type(context) == list:
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        else:
            context = self.text_embedding(context)


        # arguments
        ref_frame_idx = torch.tensor([self.ref_rope_idx] * bt, dtype=torch.long, device=t.device)
        temporal_offset = -1 if self.ref_end_mode == "replace" else 0

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            origin_seq_lens=origin_seq_lens,
            grid_sizes=grid_sizes,
            origin_grid_sizes=origin_grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            ref_frame_num=ref_frame_num,
            ref_frame_idx=ref_frame_idx,
            end_frame_num=end_frame_num,
            end_frame_idx=end_frame_idx,
            audio_embeds=audio_embeds,
            audio_attn_output_mask=audio_attn_output_mask,
            temporal_offset=temporal_offset,
            )

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                inputs = (x, e0, seq_lens, origin_seq_lens, grid_sizes, origin_grid_sizes, self.freqs, context, context_lens,
                            ref_frame_num, ref_frame_idx, end_frame_num, end_frame_idx, audio_embeds, audio_attn_output_mask, temporal_offset)

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    *inputs,
                    preserve_rng_state=True,
                    use_reentrant=False
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = x[:, :origin_seq_len, :]
        x = self.unpatchify(x, origin_grid_sizes)
        return [u.float() for u in x] if not is_input_tensor else torch.stack(x).float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
