
import os
os.environ['TORCH_HOME'] = './pretrained_models'
import random
import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed, Block
from .pos_embed import get_2d_sincos_pos_embed
from .gather_layer import GatherLayer
import torch.distributed.nn
import torch.distributed as dist
from ipdb import set_trace
from einops import rearrange, repeat
import copy
import torch.nn.functional as F
from collections import OrderedDict
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg

from .yb_tome import yb_bipartite_soft_matching


from torch.jit import Final
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
	resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked

# from kmeans_pytorch import kmeans
# from torchpq.clustering import KMeans
# from torchpq.clustering import MultiKMeans
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
		
		

# our main proposed model, for pretraining only, for finetuning, use CAVMAEFT class
class CAVMAE_BASE(nn.Module):
	""" CAV-MAE Model
	"""
	def __init__(self, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
				 embed_dim=768, modality_specific_depth=23, num_heads=16,
				 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
				 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=False,opt=None):
		super().__init__()
		print('A CAV-MAE Model')
		print('Use norm_pix_loss: ', norm_pix_loss)
		print('Learnable Positional Embedding: ', tr_pos)

		# the encoder part
		# overide the timm package
		timm.models.vision_transformer.Attention = Attention
		# timm.models.vision_transformer.PatchEmbed = PatchEmbed
		# timm.models.vision_transformer.Block = Block

		self.opt = opt

		self.vit_base = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)

		#### YB: it seems loading this ckpt geting better results.
		### Please check your path
		self.vit_base.load_state_dict(torch.load('/mnt/opr/yblin/cav-pt/src/adapt_weights/jx_vit_base_patch16_224_in21k-e5005f0a.pth'), strict=False)


		



		###### ---------> override timm
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

		
		for i in range(len(self.vit_base.blocks)):
			self.vit_base.blocks[i].norm1_a = copy.deepcopy(self.vit_base.blocks[i].norm1)
			self.vit_base.blocks[i].norm1_v = copy.deepcopy(self.vit_base.blocks[i].norm1)

			self.vit_base.blocks[i].norm2_a = copy.deepcopy(self.vit_base.blocks[i].norm2)
			self.vit_base.blocks[i].norm2_v = copy.deepcopy(self.vit_base.blocks[i].norm2)


		block_weight = OrderedDict()
		for n, p in self.vit_base.blocks.named_parameters():
			block_weight[n] = p



		self.vit_base.blocks = self.my_blocks
		self.vit_base.blocks.load_state_dict(block_weight, strict=True)

		
		

		# ##########
		self.my_patch_embed = PatchEmbed()
		self.my_patch_embed_a = PatchEmbed(in_chans=1)
		self.my_patch_embed.load_state_dict(self.vit_base.patch_embed.state_dict(), strict=True)
		self.vit_base.patch_embed = copy.deepcopy(self.my_patch_embed)


		audio_patch_weight = OrderedDict()
		audio_patch_weight['proj.weight'] = self.vit_base.patch_embed.proj.weight.mean(dim=1,keepdim=True)
		audio_patch_weight['proj.bias'] = self.vit_base.patch_embed.proj.bias
		self.my_patch_embed_a.load_state_dict(audio_patch_weight)

		
		self.vit_base.patch_embed_a = copy.deepcopy(self.my_patch_embed_a)
		self.vit_base.pos_embed_a = nn.Parameter(F.interpolate(self.vit_base.pos_embed[:,1:].permute(0,2,1),size=[512]).permute(0,2,1))
		self.vit_base.norm_a = copy.deepcopy(self.vit_base.norm)
		self.vit_base.norm_pre_a = copy.deepcopy(self.vit_base.norm_pre)


		self.ast_base = copy.deepcopy(self.vit_base)

		########### --------> new decoder
		self.mm_layer_1 = copy.deepcopy(self.vit_base.blocks[11])
		self.mm_layer_2 = copy.deepcopy(self.vit_base.blocks[11])



		self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
		self.decoder_pos_embed_a = nn.Parameter(torch.zeros(1, self.vit_base.pos_embed_a.size(1), decoder_embed_dim))  # fixed sin-cos embedding
		self.decoder_pos_embed_v = nn.Parameter(torch.zeros(1, self.vit_base.pos_embed.size(1)-1, decoder_embed_dim))  # fixed sin-cos embedding
		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

		self.decoder_blocks = nn.Sequential(*[
			Block(
				dim=512,
				num_heads=16,
				mlp_ratio=4.0,
				qkv_bias=True,
				qk_norm=False,
				init_values=None,
				proj_drop=0.0,
				attn_drop=0.0,
				drop_path=0.0,
				mlp_layer=timm.layers.mlp.Mlp,
			)
			for i in range(8)])

		self.decoder_norm = norm_layer(decoder_embed_dim)

		# project channel is different for two modality, use two projection head
		self.decoder_pred_a = nn.Linear(decoder_embed_dim, patch_size ** 2 * 1, bias=True)  # decoder to patch
		self.decoder_pred_v = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
		self.decoder_modality_a = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
		self.decoder_modality_v = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))





	def patchify(self, imgs, c, h, w, p=16):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		x = imgs.contiguous().reshape(shape=(imgs.shape[0], c, h, p, w, p))
		x = torch.einsum('nchpwq->nhwpqc', x)
		x = x.contiguous().reshape(shape=(imgs.shape[0], h * w, p ** 2 * c))
		return x

	def unpatchify(self, x, c, h, w, p=16):
		"""
		x: (N, L, patch_size**2 *3)
		imgs: (N, 3, H, W)
		"""
		assert h * w == x.shape[1]

		x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
		return imgs

	def random_masking_unstructured(self, x, mask_ratio):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - mask_ratio))

		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

		# sort noise for each sample
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		return x_masked, mask, ids_restore

	def random_masking_structured(self, x, mask_ratio, t=64, f=8, mode='time'):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - mask_ratio))

		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
		assert L == f * t
		noise = noise.reshape(N, f, t) # the audio patch is in shape [f,t], not [t,f]
		if mode == 'time':
			for i in range(N):
				mask_t_list = random.sample(range(t), int(t * mask_ratio))
				for k in mask_t_list:
					noise[i, :, k] = 1.1  # large value will be removed
		elif mode == 'freq':
			for i in range(N):
				mask_f_list = random.sample(range(f), int(f * mask_ratio))
				for k in mask_f_list:
					noise[i, k, :] = 1.1  # large value will be removed
		elif mode == 'tf':
			for i in range(N):
				mask_t_list = random.sample(range(t), int(t * mask_ratio * 0.7))
				for k in mask_t_list:
					noise[i, :, k] = 1.1  # large value will be removed
			for i in range(N):
				mask_f_list = random.sample(range(f), int(f * mask_ratio * 0.7))
				for k in mask_f_list:
					noise[i, k, :] = 1.1  # large value will be removed
		noise = noise.reshape(N, L)

		# sort noise for each sample, only need to manuplate these two ids_shuffle, ids_restore
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		return x_masked, mask, ids_restore

	def forward_encoder(self, a, v, mask_ratio_a, mask_ratio_v, no_grad_modality=None, mask_mode='unstructured'):

		# embed patches self.vit_large
		a = a.unsqueeze(1)
		a = a.transpose(2, 3)


		a = self.vit_base.patch_embed_a(a)
		a = a + self.vit_base.pos_embed_a
		a = a + self.vit_base.norm_pre_a(a)


		v = self.vit_base.patch_embed(v)
		v = v + self.vit_base.pos_embed[:,1:]
		v = v + self.vit_base.norm_pre(v)






		bs = v.size(0)


		perm = torch.randperm(bs, device=v.device)
		chunk_idx = torch.chunk(perm, 5, dim=0)


		perm = torch.randperm(bs, device=v.device)
		chunk_idx_v = torch.chunk(perm, 5, dim=0)





		a, mask_a, ids_restore_a = self.random_masking_unstructured(a, mask_ratio_a)
		v, mask_v, ids_restore_v = self.random_masking_unstructured(v, mask_ratio_v)




		# v = v.repeat((1,4,1))
		for idx_layer, blk in enumerate(self.vit_base.blocks): 


			
			v = blk(v,'v')
			# a = blk(a,'a')
			a = self.ast_base.blocks[idx_layer](a)


		cv = self.vit_base.norm(v)
		# ca = self.vit_base.norm_a(a)

		ca = self.ast_base.norm_a(a)







		x = torch.cat((ca,cv),dim=1)
		return x, mask_a, ids_restore_a, mask_v, ids_restore_v, ca, cv
	
		# return None, None, None, None, None, ca, cv

	def forward_encoder_mmixed(self, a, v, mask_ratio_a, mask_ratio_v, no_grad_modality=None, mask_mode='unstructured'):

		# embed patches self.vit_large
		a = a.unsqueeze(1)
		a = a.transpose(2, 3)


		a = self.vit_base.patch_embed_a(a)
		a = a + self.vit_base.pos_embed_a
		a = a + self.vit_base.norm_pre_a(a)


		v = self.vit_base.patch_embed(v)
		v = v + self.vit_base.pos_embed[:,1:]
		v = v + self.vit_base.norm_pre(v)






		bs = v.size(0)



		perm = torch.randperm(bs, device=v.device)
		chunk_idx = torch.chunk(perm, 5, dim=0)


		perm = torch.randperm(bs, device=v.device)
		chunk_idx_v = torch.chunk(perm, 5, dim=0)




		a_masked = []
		v_masked = []
		for batch_id, tmp_idx in enumerate(chunk_idx):
			a_tmp, mask_a, ids_restore_a = self.random_masking_structured(a[tmp_idx], 0 + 0.2*batch_id, t=64, f=8, mode='tf')
			a_masked.append(a_tmp)
	
			v_tmp, _, _ = self.random_masking_unstructured(v[chunk_idx_v[batch_id]], 0 + 0.2*batch_id)
			v_masked.append(v_tmp)



		for idx_layer, blk in enumerate(self.vit_base.blocks): 
			for idx in range(len(a_masked)):

				a_masked[idx] = self.vit_base.blocks[idx_layer](a_masked[idx],'a')
				v_masked[idx]= blk(v_masked[idx],'v')


		
		for idx in range(len(a_masked)):
			v_masked[idx] = self.vit_base.norm(v_masked[idx]).mean(dim=1,keepdim=True)


			a_masked[idx] =  self.vit_base.norm_a(a_masked[idx]).mean(dim=1,keepdim=True)




		cv = torch.cat(v_masked, dim=0)
		ca = torch.cat(a_masked, dim=0)


		a_idx = torch.cat(chunk_idx, dim=0)
		a_idx_reverse = torch.zeros_like(a_idx, dtype=torch.int64, device=v.device)


		v_idx = torch.cat(chunk_idx_v, dim=0)
		v_idx_reverse = torch.zeros_like(v_idx, dtype=torch.int64, device=v.device)


		
		for idx in range(len(a_idx)):
			v_idx_reverse[v_idx[idx]] = idx
			a_idx_reverse[a_idx[idx]] = idx


		cv = torch.gather(cv, dim=0, index=v_idx_reverse.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, cv.size(-1)))
		ca = torch.gather(ca, dim=0, index=a_idx_reverse.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, ca.size(-1)))


	
		return None, None, None, None, None, ca, cv


	def forward_decoder(self, x, mask_a, ids_restore_a, mask_v, ids_restore_v):
		
		num_patches_a = self.vit_base.pos_embed_a.size(1)
		x = self.decoder_embed(x)

		# append mask tokens to sequence
		# mask_tokens_a in shape [B, #a_mask_token, mask_token_dim], get the number of masked samples from mask_a[0], which is the first example of the batch, all samples should have same number of masked tokens
		mask_tokens_a = self.mask_token.repeat(x.shape[0], int(mask_a[0].sum()), 1)

		a_ = torch.cat([x[:, :num_patches_a-int(mask_a[0].sum()), :], mask_tokens_a], dim=1) # no cls token
		a_ = torch.gather(a_, dim=1, index=ids_restore_a.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle

		# # similar for the visual modality
		mask_tokens_v = self.mask_token.repeat(x.shape[0], int(mask_v[0].sum()), 1)
		v_ = torch.cat([x[:, num_patches_a-int(mask_a[0].sum()):, :], mask_tokens_v], dim=1) # no cls token
		v_ = torch.gather(v_, dim=1, index=ids_restore_v.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle

		# # concatenate audio and visual tokens
		a_ = a_ + self.decoder_pos_embed_a
		v_ = v_ + self.decoder_pos_embed_v
		x = torch.cat([a_, v_], dim=1)

		# decoder_pos_embed = torch.cat([self.decoder_pos_embed_a, self.decoder_pos_embed_v], dim=1)
		# x = x + decoder_pos_embed

		

		# add modality indication tokens
		x[:, 0:num_patches_a, :] = x[:, 0:num_patches_a, :] + self.decoder_modality_a
		x[:, num_patches_a:, :] = x[:, num_patches_a:, :] + self.decoder_modality_v

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x_a = self.decoder_pred_a(x[:, :num_patches_a, :])
		x_v = self.decoder_pred_v(x[:, num_patches_a:, :])

		# return audio and video tokens
		return x_a, x_v


	def forward_contrastive(self, audio_rep, video_rep, bidirect_contrast=False):
		# calculate nce loss for mean-visual representation and mean-audio representation

		audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
		video_rep = torch.nn.functional.normalize(video_rep, dim=-1)

		total = torch.mm(audio_rep, torch.transpose(video_rep, 0, 1)) / 0.05

		# by default we use single directional
		if bidirect_contrast == False:
			nce = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
			c_acc = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
			return nce, c_acc
		else:
			nce_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total, dim=0)))
			nce_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total.t(), dim=0)))
			c_acc_1 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total, dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
			c_acc_2 = torch.sum(torch.eq(torch.argmax(torch.nn.functional.softmax(total.t(), dim=0), dim=0), torch.arange(0, total.shape[0], device=audio_rep.device))) / total.shape[0]
			nce = (nce_1 + nce_2) / 2
			c_acc = (c_acc_1 + c_acc_2) / 2
			return nce, c_acc

	def forward_mae_loss(self, input, pred, mask, modality):
		if modality == 'a':
			# for audio, need to adjust the shape
			input = input.unsqueeze(1)
			input = input.transpose(2, 3)
			target = self.patchify(input, 1, int(input.shape[2]/self.vit_base.patch_embed_a.patch_size[0]), int(input.shape[3]/self.vit_base.patch_embed_a.patch_size[1]), 16)
		elif modality == 'v':
			target = self.patchify(input, 3, int(input.shape[2]/self.vit_base.patch_embed.patch_size[0]), int(input.shape[3]/self.vit_base.patch_embed.patch_size[1]), 16)

		# patch-wise normalization might minorly improve the classification performance, but will make the model lose inpainting function
		# if self.norm_pix_loss:
		# 	mean = target.mean(dim=-1, keepdim=True)
		# 	var = target.var(dim=-1, keepdim=True)
		# 	target = (target - mean) / (var + 1.e-6) ** .5
		### -------->

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
		return loss

	def forward(self, audio, imgs, mask_ratio_a=0.75, mask_ratio_v=0.75, mae_loss_weight=1., contrast_loss_weight=0.01, mask_mode='unstructured'):
		# latent is used for reconstruction (mae), latent_c_{a,v} are used for contrastive learning


		# latent, mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)


		# x , mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder(audio, imgs, 0.75, 0.75, mask_mode=mask_mode)
		# if mae loss is used
		if mae_loss_weight != 0 :
			# with torch.no_grad():
			x , mask_a, ids_restore_a, mask_v, ids_restore_v, no_latent_c_a, no_latent_c_v = self.forward_encoder(audio, imgs, 0.75, 0.75, mask_mode=mask_mode)


			x = self.mm_layer_1(x,'a')
			x = self.mm_layer_2(x,'a')

			# latent = torch.cat((mma,mmv), dim=1)
			pred_a, pred_v = self.forward_decoder(x, mask_a, ids_restore_a, mask_v, ids_restore_v)

			loss_mae_a = self.forward_mae_loss(audio, pred_a, mask_a, 'a')
			loss_mae_v = self.forward_mae_loss(imgs, pred_v, mask_v, 'v')
			loss_mae = loss_mae_a + loss_mae_v

		else:
			loss_mae_a, loss_mae_v, loss_mae = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

		# if contrastive loss is used
		if contrast_loss_weight != 0:
			# note this is single directional
			
			######### -------------> YB: for DDP
			
			# bs = latent_c_a.size(0)
			
			# latent_c_a, latent_c_a_ori = latent_c_a
			# latent_c_v, latent_c_v_ori = latent_c_v
			_ , mask_a, ids_restore_a, mask_v, ids_restore_v, latent_c_a, latent_c_v = self.forward_encoder_mmixed(audio, imgs, mask_ratio_a, mask_ratio_v, mask_mode=mask_mode)

			latent_c_a = torch.cat(GatherLayer.apply(latent_c_a), dim=0)
			latent_c_v = torch.cat(GatherLayer.apply(latent_c_v), dim=0)


			
			loss_c, c_acc = self.forward_contrastive(latent_c_a.mean(dim=1), latent_c_v.mean(dim=1), bidirect_contrast=True)
			# loss_c_2, c_acc_2 = self.forward_contrastive(latent_c_v.mean(dim=1), latent_c_v_ori.mean(dim=1),bidirect_contrast=False)

			# loss_c = 0.1*loss_c_1 + 0.1* loss_c_2 + 0.8*loss_c_3
			# c_acc = 0.1* c_acc_1 + 0.1* c_acc_2 + 0.8* c_acc_3

			loss_c = contrast_loss_weight * loss_c
		else:
			loss_c, c_acc = torch.tensor(0.0, device=audio.device), torch.tensor(0.0, device=audio.device)

		loss = loss_c + loss_mae

		return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc


# the finetuned CAV-MAE model
class CAVMAEFT_BASE(nn.Module):
	def __init__(self, label_dim, img_size=224, audio_length=1024, patch_size=16, in_chans=3,
				 embed_dim=768, modality_specific_depth=23, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, tr_pos=True):
		super().__init__()

		self.vit_base = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
		# self.vit_base = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True)

		###### ---------> override timm
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
		
		for i in range(len(self.vit_base.blocks)):
			self.vit_base.blocks[i].norm1_a = copy.deepcopy(self.vit_base.blocks[i].norm1)
			self.vit_base.blocks[i].norm1_v = copy.deepcopy(self.vit_base.blocks[i].norm1)

			self.vit_base.blocks[i].norm2_a = copy.deepcopy(self.vit_base.blocks[i].norm2)
			self.vit_base.blocks[i].norm2_v = copy.deepcopy(self.vit_base.blocks[i].norm2)


		block_weight = OrderedDict()
		for n, p in self.vit_base.blocks.named_parameters():
			block_weight[n] = p


		# self.my_blocks.load_state_dict(self.vit_large.blocks.state_dict(), strict=True)
		self.vit_base.blocks = self.my_blocks
		self.vit_base.blocks.load_state_dict(block_weight, strict=True)

		# set_trace()
		# del self.my_blocks
		
		

		# ##########
		self.my_patch_embed = PatchEmbed()
		self.my_patch_embed_a = PatchEmbed(in_chans=1)
		self.my_patch_embed.load_state_dict(self.vit_base.patch_embed.state_dict(), strict=True)
		self.vit_base.patch_embed = copy.deepcopy(self.my_patch_embed)


		audio_patch_weight = OrderedDict()
		audio_patch_weight['proj.weight'] = self.vit_base.patch_embed.proj.weight.mean(dim=1,keepdim=True)
		audio_patch_weight['proj.bias'] = self.vit_base.patch_embed.proj.bias
		self.my_patch_embed_a.load_state_dict(audio_patch_weight)

		
		self.vit_base.patch_embed_a = copy.deepcopy(self.my_patch_embed_a)
		self.vit_base.pos_embed_a = nn.Parameter(F.interpolate(self.vit_base.pos_embed[:,1:].permute(0,2,1),size=[512]).permute(0,2,1))
		self.vit_base.norm_a = copy.deepcopy(self.vit_base.norm)
		self.vit_base.norm_pre_a = copy.deepcopy(self.vit_base.norm_pre)
		
		
		# self.vit_base = copy.deepcopy(self.vit_base)
		### <---------- override timm

		self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))
		self.mlp_head_a = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))
		self.mlp_head_mm = nn.Sequential(nn.LayerNorm(embed_dim*2), nn.Linear(embed_dim*2, label_dim))
		self.mlp_head_mm_v2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, label_dim))


		self.mm_layer_1 = copy.deepcopy(self.vit_base.blocks[10])
		self.mm_layer_2 = copy.deepcopy(self.vit_base.blocks[11])
	
	## run this after loading pre-training weight
	def __create_fusion__(self):
		self.mm_layer_1 = copy.deepcopy(self.vit_base.blocks[10])
		self.mm_layer_2 = copy.deepcopy(self.vit_base.blocks[11])
		
	def forward(self, a, v, mode, is_eval=False):
		if mode == 'audioonly':

			a = a.unsqueeze(1)
			a = a.transpose(2, 3)
			a = self.vit_base.patch_embed_a(a)
			a = a + self.vit_base.pos_embed_a
			a = a + self.vit_base.norm_pre_a(a)

			for blk in self.vit_base.blocks:  ### YB: !!! important
				a = blk(a,'a')

	
			a = self.vit_base.norm_a(a)
			x = a.mean(dim=1)
			out_a = self.mlp_head_a(x)


			if is_eval:
				out_a = out_a.unsqueeze(1)


			return out_a

		# finetune with only image (and inference with only audio when the model is finetuned with only image)
		elif mode == 'videoonly':
			

			bs = v.size(0)
			t = v.size(1)
			v = rearrange(v, 'b t c w h -> (b t) c w h')


			# v = self.vit_base.forward_features(v)




			v = self.vit_base.patch_embed(v)
			v = v + self.vit_base.pos_embed[:,1:]
			v = v + self.vit_base.norm_pre(v)
			for blk in self.vit_base.blocks:
				v = blk(v, 'v')


			v = self.vit_base.norm(v)


			x = v.mean(dim=1)
			x = self.mlp_head(x)

			x = rearrange(x, '(b t) p-> b t p', b=bs , t=t).squeeze(1)


			return x
			


		elif mode == 'retrieval':

			a = a.unsqueeze(1)
			a = a.transpose(2, 3)
			a = self.vit_base.patch_embed_a(a)
			a = a + self.vit_base.pos_embed_a
			a = a + self.vit_base.norm_pre_a(a)

			for blk in self.vit_base.blocks: 
				a = blk(a,'a')

	
			a = self.vit_base.norm_a(a)
			# a = a.mean(dim=1)

			################
			bs = v.size(0)
			t = v.size(1)
			v = rearrange(v, 'b t c w h -> (b t) c w h')



			v = self.vit_base.patch_embed(v)
			v = v + self.vit_base.pos_embed[:,1:]
			v = v + self.vit_base.norm_pre(v)

			for blk in self.vit_base.blocks: ### YB: !!! important
				v = blk(v, 'v')

			v = self.vit_base.norm(v)


			v = rearrange(v, '(b t) p d-> b t p d', b=bs , t=t)
			

			return a, v[:,5]


		elif mode == 'mm_grad':
	
			if is_eval:
				a = a.unsqueeze(1)
				a = a.transpose(2, 3)
				a = self.vit_base.patch_embed_a(a)
				a = a + self.vit_base.pos_embed_a
				a = a + self.vit_base.norm_pre_a(a)

				for blk in self.vit_base.blocks: 
					a = blk(a,'a')

		
				a = self.vit_base.norm_a(a)
				# a = a.mean(dim=1)

				################
				bs = v.size(0)
				t = v.size(1)
				v = rearrange(v, 'b t c w h -> (b t) c w h')



				v = self.vit_base.patch_embed(v)
				v = v + self.vit_base.pos_embed[:,1:]
				v = v + self.vit_base.norm_pre(v)

				for blk in self.vit_base.blocks: ### YB: !!! important
					v = blk(v, 'v')

				v = self.vit_base.norm(v)


				v = rearrange(v, '(b t) p d-> b t p d', b=bs , t=t)
				


				all_out = []
				for t_idx in range(10):
					av = torch.cat((a,v[:,t_idx]), dim=1)	
					av = self.mm_layer_1(av, 'a')
					av = self.mm_layer_2(av, 'a')


					av = torch.cat((
						av[:,:512].mean(dim=1),
						av[:,512:].mean(dim=1)
					), dim=-1)

					# out = self.mlp_head_mm_v2 (av.mean(dim=1)).unsqueeze(1)
					out = self.mlp_head_mm(av).unsqueeze(1)
					


					all_out.append(out)
				all_out = torch.hstack(all_out)

				return all_out

			################### ------------> training <--------------- ########
			else:
				# with torch.no_grad():
				a = a.unsqueeze(1)
				a = a.transpose(2, 3)
				a = self.vit_base.patch_embed_a(a)
				a = a + self.vit_base.pos_embed_a
				a = a + self.vit_base.norm_pre_a(a)

				for blk in self.vit_base.blocks: 
					a = blk(a,'a')

		
				a = self.vit_base.norm_a(a)


				################
				bs = v.size(0)
				t = v.size(1)
				v = rearrange(v, 'b t c w h -> (b t) c w h')



				v = self.vit_base.patch_embed(v)
				v = v + self.vit_base.pos_embed[:,1:]
				v = v + self.vit_base.norm_pre(v)

				for blk in self.vit_base.blocks: ### YB: !!! important
					v = blk(v, 'v')

				v = self.vit_base.norm(v)



				out_a = self.mlp_head_a(a.mean(dim=1))
				out_v = self.mlp_head(v.mean(dim=1))

				av = torch.cat((a,v), dim=1)
				av = self.mm_layer_1(av, 'a')
				av = self.mm_layer_2(av, 'a')
				


				av = torch.cat((
					av[:,:512].mean(dim=1),
					av[:,512:].mean(dim=1)
				), dim=-1)


				# out = self.mlp_head_mm_v2 (av.mean(dim=1))
				out = self.mlp_head_mm(av)#.unsqueeze(1)
				

				
				return out, out_a, out_v