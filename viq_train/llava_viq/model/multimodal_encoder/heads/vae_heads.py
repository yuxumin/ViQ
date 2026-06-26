"""VAE decoder heads for the viq vision encoder.

Split out of ``siglip_vit_anyres_viq`` to keep that file focused on the
encoder itself. Selected at runtime by ``MOVQ_TYPE`` in the encoder builder.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from omegaconf import OmegaConf

from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

try:
    from llava_viq._paths import PROJECT_ROOT as _PROJECT_ROOT
except ImportError:
    _PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
    )

# ldm AutoencoderKL (used by FixVAEHead / FixVAEHead_F16)
from ..vae.ldm.vae import AutoencoderKL
from ..envir_defines import *
from ..vae.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from ..losses.perceptual_loss import PerceptualLoss


class FixVAEHead(nn.Module):
    def __init__(self, in_dims, vae_path):
        super().__init__()

        self.conv_norm_out = nn.GroupNorm(num_channels=in_dims, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dims, 4 * 16, 3, padding=1)
        self.apply(self._init_weights)

        print(f'init vae from {vae_path}')
        self.vae = AutoencoderKL.from_pretrained(vae_path)
        self.vae.requires_grad_(False)    
        self.vae.eval()
        self.register_buffer('global_step', torch.tensor(0), persistent = True)

    @torch.no_grad()
    def forward(self, image):
        mean = self.vae.encode(image).latent_dist.mean
        output = (mean - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        output = output.clone().detach()
        return output
    
    @torch.no_grad()
    def forward_dist(self, image):
        parameters = self.vae.encode(image).latent_dist.parameters.detach()
        return DiagonalGaussianDistribution(parameters)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()

    def recon(self, feat):
        latents = feat
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0] # B 3 H W
        image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8) # H W 3
        return image_numpy

    def forward_loss_varlen(self, input_feat, input_images, slen, cu_slens, image_sizes, num_prefix_tokens=0, plot=True):
        feature_list = input_feat.split(slen, dim=1)
        loss = 0.0
        num_images = 0
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            B, _, H, W = image.shape

            if (H == W and H < 256):
                continue

            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W

            B, C, H ,W = shaped_feat.shape
            shaped_feat = shaped_feat.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)

            # mean
            pred_mean = shaped_feat
            target_dist = self.forward_dist(image) # b 16 w/8 h/8

            # dist
            
            dist_mse_loss = torch.mean((pred_mean - target_dist.mean.detach()) ** 2)

            dist_nll_loss = target_dist.nll(pred_mean) * 1e-14
            
            

            _this_loss = torch.mean(dist_mse_loss + dist_nll_loss)
            loss += _this_loss

            self.global_step += 1
            num_images += 1

            if self.global_step % 1000 == 0 and dist.get_rank() == 0:
                shaped_feat = pred_mean
                shaped_feat = (shaped_feat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                recon_image_numpy = self.recon(shaped_feat)
                
                shaped_feat = target_dist.sample()
                shaped_feat = (shaped_feat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                target_image_numpy = self.recon(shaped_feat)
                
                combined = np.hstack([recon_image_numpy, target_image_numpy])
                save_dir = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plot', THIS_EXP_NAME)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
                image = Image.fromarray(combined, mode='RGB')
                image.save(save_path)

        if num_images == 0:
            return torch.tensor(0.0, device=input_feat.device)
        else:
            return loss / num_images

class QwenImageVAEHead(nn.Module):
    def __init__(self, in_dims, vae_path, factor=2):
        super().__init__()

        self.conv_norm_out = nn.GroupNorm(num_channels=in_dims, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dims, factor * factor * 16, 3, padding=1)
        self.apply(self._init_weights)
        self.vae_path = vae_path
        self.factor = factor

        print(f'init vae from {vae_path}. With factor {self.factor}')

        self.vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path
        )
        self.vae.requires_grad_(False)    
        self.vae.eval()
        self.register_buffer('global_step', torch.tensor(0), persistent = True)
        
    
    @torch.no_grad()
    def forward_dist(self, image):
        parameters = self.vae.encode(image).latent_dist.parameters.detach()
        return DiagonalGaussianDistribution(parameters)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()

    def recon(self, feat):
        if USE_CPU_VIS:
            if not hasattr(self, 'vae_cpu'):
                self.vae_cpu = AutoencoderKLQwenImage.from_pretrained(self.vae_path).bfloat16().cpu()
            latents = feat.bfloat16().cpu()
            image = self.vae_cpu.decode(latents, return_dict=False)[0][:, :, 0] # B 3 H W
            image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
        else:
            latents = feat.bfloat16()
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0] # B 3 H W
            image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
        return image_numpy

    def forward_recon_list(self, input_feat, slen, cu_slens, image_sizes, num_prefix_tokens=0):
        feature_list = input_feat.split(slen, dim=1)
        recon_image_numpy_list = []
        for feat, _image_size in zip(feature_list, image_sizes):
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W

            B, C, H ,W = shaped_feat.shape
            shaped_feat = shaped_feat.reshape(B, C//(self.factor**2), self.factor, self.factor, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//(self.factor**2), H*self.factor, W*self.factor)

            # mean
            pred_mean = shaped_feat.unsqueeze(2)

            
            shaped_feat = pred_mean
            recon_image_numpy = self.recon(shaped_feat)
            recon_image_numpy_list.append(recon_image_numpy)
                
        return recon_image_numpy_list

    def forward_loss_varlen(self, input_feat, input_images, slen, cu_slens, image_sizes, num_prefix_tokens=0, plot=True):
        feature_list = input_feat.split(slen, dim=1)
        loss = 0.0
        num_images = 0
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            B, _, H, W = image.shape
            ori_H, ori_W = H, W

            if (H == W and H < 256):
                continue
        
            if LOWVRAM_MODE:
                if num_images >= INFERENCE_IMAGE_SIZE:
                    continue

            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W

            B, C, H ,W = shaped_feat.shape
            shaped_feat = shaped_feat.reshape(B, C//(self.factor**2), self.factor, self.factor, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//(self.factor**2), H*self.factor, W*self.factor)

            # mean
            pred_mean = shaped_feat.unsqueeze(2)
            target_dist = self.forward_dist(image.unsqueeze(2))  # extend num_frame to 1 # b 16 w/8 h/8

            # dist
            # TODO: 修改latent vae loss to cosine loss
            dist_mse_loss = torch.mean((pred_mean - target_dist.mean.detach()) ** 2)

            dist_nll_loss = target_dist.nll(pred_mean) * 1e-13
            
            

            _this_loss = torch.mean(dist_mse_loss + dist_nll_loss)
            loss += _this_loss

            self.global_step += 1
            num_images += 1

            if not LOWVRAM_MODE:
                if (self.global_step % 1000 == 0 or self.global_step == 1) and dist.get_rank() == 0 and ori_H < 1024 and ori_W < 1024:
                    shaped_feat = pred_mean
                    recon_image_numpy = self.recon(shaped_feat)
                    
                    shaped_feat = target_dist.sample()
                    target_image_numpy = self.recon(shaped_feat)
                    
                    combined = np.hstack([recon_image_numpy, target_image_numpy])
                    save_dir = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plot', THIS_EXP_NAME)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
                    image = Image.fromarray(combined, mode='RGB')
                    image.save(save_path)

        if num_images == 0:
            return torch.tensor(0.0, device=input_feat.device)
        else:
            return loss / num_images


class QwenImageVAEHead_trainable(nn.Module):
    def __init__(self, in_dims, vae_path):
        super().__init__()

        self.conv_norm_out = nn.GroupNorm(num_channels=in_dims, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dims, 4 * 16, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(in_dims, 4 * 16, 3, padding=1)
        self.apply(self._init_weights)
        self.vae_path = vae_path

        print(f'init vae from {vae_path}')

        self.vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path
        )

        self.perceptual_loss = PerceptualLoss('lpips').eval()
        self.perceptual_loss.requires_grad_(False)    
        self.perceptual_loss.eval()

        self.register_buffer('global_step', torch.tensor(0), persistent = True)
        # del self.vae.decoder.up_blocks[1].upsamplers[0].time_conv
        # del self.vae.decoder.up_blocks[0].upsamplers[0].time_conv
        
    
    @torch.no_grad()
    def forward_dist(self, image):
        parameters = self.vae.encode(image).latent_dist.parameters.detach()
        return DiagonalGaussianDistribution(parameters)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()

    def recon(self, feat, return_image_raw=False):
        latents = feat.bfloat16()
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0] # B 3 H W
        if return_image_raw:
            return image
        image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
        return image_numpy

    def forward_recon_list(self, input_feat, slen, cu_slens, image_sizes, num_prefix_tokens=0):
        raise
        feature_list = input_feat.split(slen, dim=1)
        recon_image_numpy_list = []
        for feat, _image_size in zip(feature_list, image_sizes):
            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B 4C H W

            B, C, H ,W = shaped_feat.shape
            shaped_feat = shaped_feat.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)

            # mean
            pred_mean = shaped_feat.unsqueeze(2)

            
            shaped_feat = pred_mean
            recon_image_numpy = self.recon(shaped_feat)
            recon_image_numpy_list.append(recon_image_numpy)
                
        return recon_image_numpy_list

    def forward_loss_varlen(self, input_feat, input_images, slen, cu_slens, image_sizes, num_prefix_tokens=0, plot=True, deterministic=False):
        feature_list = input_feat.split(slen, dim=1)
        loss = 0.0
        num_images = 0
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            ori_image = image
            B, _, H, W = image.shape
            ori_H, ori_W = H, W

            if (H == W and H < 256):
                continue
        
            if LOWVRAM_MODE:
                if num_images >= INFERENCE_IMAGE_SIZE:
                    continue

            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_act(self.conv_norm_out(shaped_feat)) # B 4C H W

            shaped_feat_mean = self.conv_out(shaped_feat) # B 4C H W
            shaped_feat_logvar = self.conv_out_logvar(shaped_feat) # B 4C H W

            B, C, H ,W = shaped_feat_mean.shape
            shaped_feat_mean = shaped_feat_mean.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2) # f8d16
            shaped_feat_logvar = shaped_feat_logvar.reshape(B, C//4, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C//4, H*2, W*2)
        
            shaped_parameters = torch.cat([shaped_feat_mean, shaped_feat_logvar], dim=1)
            posterior = DiagonalGaussianDistribution(shaped_parameters)

            if self.training and (not deterministic):
                vae_latent_feat = posterior.sample()
            else:
                vae_latent_feat = posterior.mode()

            # mean
            pred_mean = vae_latent_feat.unsqueeze(2)
            recon_image = self.recon(pred_mean, return_image_raw=True)

            # loss
            mse_loss = F.mse_loss(ori_image, recon_image, reduction="mean")
            perceptual_loss = self.perceptual_loss(ori_image, recon_image).mean()
            kl_loss = posterior.kl()
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

            _this_loss = 1e-6 * kl_loss + mse_loss + perceptual_loss
            loss += _this_loss

            self.global_step += 1
            num_images += 1

            if not LOWVRAM_MODE:
                if (self.global_step % 1000 == 0 or self.global_step == 1) and dist.get_rank() == 0 and ori_H < 1024 and ori_W < 1024:
                    shaped_feat = pred_mean
                    recon_image_numpy = self.recon(shaped_feat)
                    
                    target_image_numpy = ((ori_image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
                    
                    combined = np.hstack([recon_image_numpy, target_image_numpy])
                    save_dir = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plotv2', THIS_EXP_NAME)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
                    image = Image.fromarray(combined, mode='RGB')
                    image.save(save_path)

        if num_images == 0:
            return torch.tensor(0.0, device=input_feat.device)
        else:
            return loss / num_images

class FixVAEHead_F16(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.conv_norm_out = nn.GroupNorm(num_channels=in_dims, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(in_dims, 16, 3, padding=1)
        self.apply(self._init_weights)

        _ldm_dir = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'model', 'multimodal_encoder', 'vae', 'ldm')
        _ldm_ckpt = os.path.join(_ldm_dir, 'model.ckpt')
        _ldm_config = os.path.join(_ldm_dir, 'config.yaml')
        print(f'init vae from {_ldm_ckpt}')


        config = OmegaConf.load(_ldm_config)
        self.vae = AutoencoderKL(
            ddconfig = config.model.params.ddconfig,
            embed_dim = config.model.params.embed_dim,
            ckpt_path = _ldm_ckpt
        )
        
        self.vae.requires_grad_(False)    
        self.vae.eval()
        self.register_buffer('global_step', torch.tensor(0), persistent = True)
    
    @torch.no_grad()
    def forward_dist(self, image):
        parameters = self.vae.encode(image).parameters.detach()
        return DiagonalGaussianDistribution(parameters)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            m.bias.data.zero_()

    def recon(self, feat):
        latents = feat
        image = self.vae.decode(latents) # B 3 H W
        image_numpy = ((image[0].permute(1, 2, 0).float().detach().cpu() * 0.5 + 0.5).clamp(0, 1).numpy() * 255).astype(np.uint8) # H W 3
        return image_numpy

    def forward_loss_varlen(self, input_feat, input_images, slen, cu_slens, image_sizes, num_prefix_tokens=0, plot=True):
        feature_list = input_feat.split(slen, dim=1)
        loss = 0.0
        num_images = 0
        for feat, image, _image_size in zip(feature_list, input_images, image_sizes):
            B, _, H, W = image.shape

            if (H == W and H < 256):
                continue

            shaped_feat = feat[:, num_prefix_tokens:].reshape(len(feat), _image_size[0], _image_size[1], -1).permute(0, 3, 1, 2) # B C H W
            shaped_feat = self.conv_out(self.conv_act(self.conv_norm_out(shaped_feat))) # B C H W

            # mean
            pred_mean = shaped_feat
            target_dist = self.forward_dist(image) # b 16 w/8 h/8

            # dist
            
            dist_mse_loss = torch.mean((pred_mean - target_dist.mean.detach()) ** 2)
            dist_nll_loss = target_dist.nll(pred_mean) * 1e-14

            
            

            _this_loss = torch.mean(dist_mse_loss + dist_nll_loss)
            loss += _this_loss

            self.global_step += 1
            num_images += 1

            if self.global_step % 1000 == 0 and dist.get_rank() == 0 and H < 1024 and W < 1024:
                shaped_feat = pred_mean
                recon_image_numpy = self.recon(shaped_feat)
                
                shaped_feat = target_dist.sample()
                target_image_numpy = self.recon(shaped_feat)
                
                combined = np.hstack([recon_image_numpy, target_image_numpy])
                save_dir = os.path.join(str(_PROJECT_ROOT), 'llava_viq', 'training_plot', THIS_EXP_NAME)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"image_{self.global_step}.png")
                image = Image.fromarray(combined, mode='RGB')
                image.save(save_path)

        if num_images == 0:
            return torch.tensor(0.0, device=input_feat.device)
        else:
            return loss / num_images

