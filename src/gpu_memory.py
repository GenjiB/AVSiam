import argparse

import torch
from torch import nn
from transformers import ViTModel
import timm

from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
import torch.distributed.nn
import torch.distributed as dist
from ipdb import set_trace
from einops import rearrange, repeat
import copy
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument("--share_encoder", action="store_true")
parser.add_argument("--model_name", type=str, default="vit_h")

args = parser.parse_args()

if args.model_name == "vit_h":
	args.model_name = "google/vit-huge-patch14-224-in21k"
elif args.model_name == "vit_l":
	args.model_name = "google/vit-large-patch16-224"
elif args.model_name == "vit_b":
	args.model_name = "google/vit-base-patch16-224-in21k"
else:
	raise ValueError("not supported model name: {args.model_name}")

torch.cuda.set_device("cuda:0")


def max_mem():
	"""max gpu memory allocated in GB"""
	mem = torch.cuda.max_memory_allocated("cuda:0") / 1024 / 1024 / 1024
	return mem


def get_num_params(model):
	return sum(p.numel() for p in model.parameters())


from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
	resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from torch.jit import Final
class Attention(nn.Module):
	fused_attn: Final[bool]
	def __init__(
			self,
			dim,
			num_heads=8,
			qkv_bias=False,
			qk_norm=False,
			attn_drop=0.,
			proj_drop=0.,
			norm_layer=nn.LayerNorm,
	):
		super().__init__()
		assert dim % num_heads == 0, 'dim should be divisible by num_heads'
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		self.scale = self.head_dim ** -0.5
		self.fused_attn = use_fused_attn()

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x, tome=False):
		B, N, C = x.shape
		qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = self.q_norm(q), self.k_norm(k)

		if self.fused_attn:
			x = F.scaled_dot_product_attention(
				q, k, v,
				dropout_p=self.attn_drop.p,
			)
		else:
			q = q * self.scale
			attn = q @ k.transpose(-2, -1)
			attn = attn.softmax(dim=-1)
			attn = self.attn_drop(attn)
			x = attn @ v

		x = x.transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)

		if tome:
			return x, k.mean(1)
		else:
			return x

class PatchEmbed(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
		super().__init__()

		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
		self.img_size = img_size
		self.patch_size = patch_size
		self.num_patches = num_patches

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		x = self.proj(x).flatten(2).transpose(1, 2)
		return x

class Block(nn.Module):

	def __init__(
			self,
			dim,
			num_heads,
			mlp_ratio=4.,
			qkv_bias=False,
			qk_norm=False,
			proj_drop=0.,
			attn_drop=0.,
			init_values=None,
			drop_path=0.,
			act_layer=nn.GELU,
			norm_layer=nn.LayerNorm,
			mlp_layer=Mlp,
	):
		super().__init__()
		self.norm1 = norm_layer(dim)
		self.norm1_a = norm_layer(dim)
		self.norm1_v = norm_layer(dim)
		self.attn = Attention(
			dim,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			qk_norm=qk_norm,
			attn_drop=attn_drop,
			proj_drop=proj_drop,
			norm_layer=norm_layer,
		)
		self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

		self.norm2 = norm_layer(dim)
		self.norm2_a = norm_layer(dim)
		self.norm2_v = norm_layer(dim)
		self.mlp = mlp_layer(
			in_features=dim,
			hidden_features=int(dim * mlp_ratio),
			act_layer=act_layer,
			drop=proj_drop,
		)
		self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
		self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()



	def forward(self, x, modality=None, r=0):
		if modality == None:
			x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
			x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
			return x
		elif modality == 'a':
			# if r > 0:
			# 	x_attn, x_k = self.attn(self.norm1_a(x), tome=True)
			# 	x = x + self.drop_path1(self.ls1(x_attn))
			# 	# Apply ToMe here
			# 	merge, _ = bipartite_soft_matching(
			# 		x_k,
			# 		r,
			# 		class_token=False,
			# 		distill_token=False,
			# 	)
			# 	x, _ = merge_wavg(merge, x)
			# 	x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a(x))))

			# else:
			x = x + self.drop_path1(self.ls1(self.attn(self.norm1_a(x))))
			x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_a(x))))

			return x
		elif modality == 'v':

			# if r > 0:
			# 	x_attn, x_k = self.attn(self.norm1_v(x), tome=True)
			# 	x = x + self.drop_path1(self.ls1(x_attn))
			# 	# Apply ToMe here
			# 	merge, _ = bipartite_soft_matching(
			# 		x_k,
			# 		r,
			# 		class_token=False,
			# 		distill_token=False,
			# 	)

			# 	x, _ = merge_wavg(merge, x)
			# 	x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_v(x))))

			# else:
			x = x + self.drop_path1(self.ls1(self.attn(self.norm1_v(x))))
			x = x + self.drop_path2(self.ls2(self.mlp(self.norm2_v(x))))

			return x

		elif modality == 'av': # input: [a,v] #a: BxNxD
			a,v = x

			num_a = a.size(1)
			a = self.norm1_a(a)
			v = self.norm1_v(v)

			x = torch.cat((a,v), dim=1)
			x = x + self.drop_path1(self.ls1(self.attn(x)))
			
			a2 = self.norm2_a(x[:, :num_a])
			v2 = self.norm2_v(x[:, num_a:])

			x2 = torch.cat((a2,v2), dim=1)

			out = x + self.drop_path2(self.ls2(self.mlp(x2)))
			return out[:, :num_a], x[:, num_a:]
		
		

class PatchEmbed(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
		super().__init__()

		img_size = to_2tuple(img_size)
		patch_size = to_2tuple(patch_size)
		num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
		self.img_size = img_size
		self.patch_size = patch_size
		self.num_patches = num_patches

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		x = self.proj(x).flatten(2).transpose(1, 2)
		return x


class DummyModel(nn.Module):
	"""docstring for DummyModel"""

	def __init__(self, args):
		super().__init__()
		self.args = args

		self.vit_base = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
		
		self.my_patch_embed = PatchEmbed()
		self.my_blocks = nn.Sequential(*[
			Block(
				dim=768,
				num_heads=12,
				mlp_ratio=4.0,
				qkv_bias=True,
				qk_norm=False,
				init_values=None,
				proj_drop=0.0,
				attn_drop=0.0,
				drop_path=0.0,
				mlp_layer=timm.layers.mlp.Mlp,
			)
			for i in range(12)])
		self.vit_base.blocks = self.my_blocks


		self.mm_layer_1 = copy.deepcopy(self.vit_base.blocks[10])
		self.mm_layer_2 = copy.deepcopy(self.vit_base.blocks[11])

	def forward(self, v, a):

		
		a = self.my_patch_embed(a)

		v = self.my_patch_embed(v)
		v = v + self.vit_base.pos_embed[:,1:]


		for blk in self.vit_base.blocks: ### YB: !!! important
			a = blk(a)
			v = blk(v)

		h = torch.cat([a, v], dim=1)
		h = self.mm_layer_1(h)
		z = self.mm_layer_1(h)
		return z.mean()  # dummy loss


model = DummyModel(args)
model.cuda()

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

num_params = get_num_params(model)

print(f"=======share_encoder:{args.share_encoder}====")
print(f"=======model:{args.model_name}====")
print(f"number of parameters: {num_params/1024/1024/1024} B")
print(f"GPU mem: {max_mem()} GB")

# simulate training

bs = 4
for step in range(3):

	vid = torch.rand(bs, 3, 224, 224).cuda()  # 256 tokens
	audio = torch.rand(bs, 3, 128, 1024).cuda()  # 256 tokens
	loss = model(vid, audio)

	optim.zero_grad()
	loss.backward()
	optim.step()
	print(f"step: {step}. max gpu mem: {max_mem()} GB")