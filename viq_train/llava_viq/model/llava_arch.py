#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower, build_vision_tower_two
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava_viq.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava_viq.mm_utils import get_anyres_image_grid_shape, get_anyres_image_grid_shape_ours
import os
import torch.distributed as dist
import torch.nn.functional as F
import random

if 'ZERO_IMAGE_FORWARD' in os.environ:
    print(f"ZERO_IMAGE_FORWARD is set")
    ZERO_IMAGE_FORWARD = True
else:
    ZERO_IMAGE_FORWARD = False



if 'TRAIN_CLS_TOKEN' in os.environ:
    print(f"TRAIN_CLS_TOKEN is set")
    TRAIN_CLS_TOKEN = True
else:
    TRAIN_CLS_TOKEN = False

if 'ORIGIN_CLS_LOSS' in os.environ:
    print(f"ORIGIN_CLS_LOSS is set")
    ORIGIN_CLS_LOSS = True
else:
    ORIGIN_CLS_LOSS = False


if 'CLS_MAX_RES' in os.environ:
    CLS_MAX_RES = int(os.environ['CLS_MAX_RES'])
    print(f"CLS_MAX_RES is set to {CLS_MAX_RES}")
else:
    CLS_MAX_RES = False







# VQ_GEN_VIT is always enabled for ViQ (build the quantized vision tower path).
VQ_GEN_VIT = True


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            
        if hasattr(config, "mm_vision_tower_two") and config.mm_vision_tower_two is not None:
            self.vision_tower_two = build_vision_tower_two(config, delay_load=True)


    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vision_tower_two(self):
        vision_tower = getattr(self, 'vision_tower_two', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        vision_tower_two = model_args.vision_tower_two
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_mlp_adapter_two = model_args.pretrain_mm_mlp_adapter_two
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.mm_vision_tower_two = vision_tower_two

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()
        

        if model_args.vision_tower_two is not None:
            if self.get_vision_tower_two() is None:
                vision_tower_two = build_vision_tower_two(model_args)
                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower_two = [vision_tower_two]
                else:
                    self.vision_tower_two = vision_tower_two
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower_two = self.vision_tower_two[0]
                else:
                    vision_tower_two = self.vision_tower_two
                vision_tower_two.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(vision_resampler, 'hidden_size', vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)
            print("Building mm_projector...")
        
            if getattr(model_args, 'mm_projector_type_two', None) is not None:
                if getattr(self, 'mm_projector_two', None) is None:
                    self.mm_projector_two = build_vision_projector(self.config, vision_cfg=vision_tower_two.config)
                    print("Building mm_projector_two...")
        

        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

            if hasattr(self, 'mm_projector_two') and self.mm_projector_two is not None:
                for p in self.mm_projector_two.parameters():
                    p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            if self.config.mm_projector_type == 'multipath':
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword, replace_keyword):
                    return {replace_keyword + k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                projector1_weights = get_w(mm_projector_weights, 'mm_projector', 'vision_projector1.')

                if pretrain_mm_mlp_adapter_two is not None:
                    mm_projector_weights_two = torch.load(pretrain_mm_mlp_adapter_two, map_location='cpu')
                    projector2_weights = get_w(mm_projector_weights_two, 'mm_projector', 'vision_projector2.')
                
                # merge two weight dicts
                projector1_weights.update(projector2_weights)
                self.mm_projector.load_state_dict(projector1_weights)

            elif self.config.mm_projector_type == 'multipath_conv_mlp' or self.config.mm_projector_type == 'multipath_conv_mlp_woconv':
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword, replace_keyword):
                    return {replace_keyword + k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                projector1_weights = get_w(mm_projector_weights, 'mm_projector.proj', 'vision_projector1.')

                if pretrain_mm_mlp_adapter_two is not None:
                    mm_projector_weights_two = torch.load(pretrain_mm_mlp_adapter_two, map_location='cpu')
                    if self.config.mm_projector_type == 'multipath_conv_mlp':
                        projector2_weights = get_w(mm_projector_weights_two, 'mm_projector.proj', 'vision_projector2.')
                    else:
                        projector2_weights = get_w(mm_projector_weights_two, 'mm_projector', 'vision_projector2.')
                
                # merge two weight dicts
                projector1_weights.update(projector2_weights)
                self.mm_projector.load_state_dict(projector1_weights)
                
            else:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

                if pretrain_mm_mlp_adapter_two is not None:
                    mm_projector_weights_two = torch.load(pretrain_mm_mlp_adapter_two, map_location='cpu')
                    self.mm_projector_two.load_state_dict(get_w(mm_projector_weights_two, 'mm_projector'))

            def get_w_sampler(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            incompatible_keys = self.vision_resampler.load_state_dict(get_w_sampler(mm_projector_weights, 'vision_resampler'), strict=False)
            print(incompatible_keys)
            
                    
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_vision_tower_two(self):
        return self.get_model().get_vision_tower_two()

    def encode_images(self, images):
        img_size = None
        # eva_g_anysize_highres should always paired with conv_mlp
        if self.get_model().config.mm_vision_tower == 'eva_vit_g_anysize_anyres':
            image_features, img_size = self.get_model().get_vision_tower()(images)
        else:
            image_features = self.get_model().get_vision_tower()(images)

        if self.get_model().config.mm_projector_type == 'multipath':
            image_features = self.get_model().mm_projector.forward1(image_features)
        elif self.get_model().config.mm_projector_type.startswith('conv_mlp'):
            image_features = self.get_model().mm_projector(image_features, img_size)
        elif self.get_model().config.mm_projector_type == 'multipath_conv_mlp' or self.get_model().config.mm_projector_type == 'multipath_conv_mlp_woconv':
            image_features = self.get_model().mm_projector.forward1(image_features, img_size)
        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_images_two(self, images):
        image_features = self.get_model().get_vision_tower_two()(images)
        if type(image_features) is tuple:
            image_features, image_size = image_features
            if self.get_model().config.mm_projector_type == 'multipath_conv_mlp' or self.get_model().config.mm_projector_type == 'multipath_conv_mlp_woconv':
                image_features = self.get_model().mm_projector.forward2(image_features, image_size)
                return image_features
        if self.get_model().config.mm_projector_type == 'multipath':
            image_features = self.get_model().mm_projector.forward2(image_features)
        elif self.get_model().config.mm_projector_type == 'multipath_conv_mlp' or self.get_model().config.mm_projector_type == 'multipath_conv_mlp_woconv':
            image_features = self.get_model().mm_projector.forward2(image_features)
        else:
            image_features = self.get_model().mm_projector(image_features)
        return image_features

    def negative_cos_loss(self, pred, gt, loss_mask=None):
        pred = torch.nn.functional.normalize(pred, dim=-1)
        gt = torch.nn.functional.normalize(gt, dim=-1)
        loss = 1.0 - (pred * gt).sum(dim=-1)
        if loss_mask is not None:
            num_supervised = loss_mask.sum()
            if num_supervised < 0.5:
                loss = (loss * loss_mask).sum()
            else:
                loss = (loss * loss_mask).sum() / num_supervised
            return loss
        else:
            return loss.mean()

    def random_crop(self, images, base_size=384, patch_size=16):
        images_cropped = []
        crop_pos = []
        patch_num = base_size // patch_size # 24
        for image in images:
            # bs, _, h, w = image.shape
            
            # h, w = image.shape[-2:]


            h, w = image.shape[-2:]

            h_patch = h // patch_size
            w_patch = w // patch_size

            crop_h = h_patch - patch_num
            crop_w = w_patch - patch_num
            crop_top = torch.randint(0, crop_h + 1, (1,)).item()
            crop_left = torch.randint(0, crop_w + 1, (1,)).item()

            crop_pos.append((crop_top, crop_left, patch_num, patch_num))
            image_cropped = image[..., crop_top * patch_size: (crop_top+patch_num)*patch_size,
                                   crop_left*patch_size: (crop_left+patch_num)*patch_size]
            images_cropped.append(image_cropped)
        
        images_cropped = torch.cat(images_cropped, dim=0)
        return images_cropped, crop_pos

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, highres_images=None, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        zero_feature = 0
        if type(images) is list:
            num_images = len(images)
            world_size = dist.get_world_size()
            tensor_in = torch.zeros(1,  dtype=torch.int64, device=images[0].device).fill_(num_images)
            tensor_out = torch.zeros(world_size, dtype=torch.int64, device=images[0].device)
            dist.all_gather_into_tensor(tensor_out, tensor_in)
            max_num_images = tensor_out.max().item()


            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            if max_num_images > num_images:
                aimg = images[-1]
                for _ in range(max_num_images - num_images):
                    images.append(aimg.new(1, 3, 64, 64).fill_(0))

            if TRAIN_CLS_TOKEN:
                if VQ_GEN_VIT:
                    raw_image_features, img_sizes, cls_token, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code = self.get_model().get_vision_tower()(images, cal_attn_pool=True)
                else:
                    raw_image_features, img_sizes, cls_token = self.get_model().get_vision_tower()(images, cal_attn_pool=True)
                _, _, cls_token_teacher = self.get_model().get_vision_tower().forward_teacher(images)
                loss_mask = torch.zeros(cls_token.shape[0], dtype=cls_token.dtype, device=cls_token.device)
                loss_mask[:num_images] = 1
                if CLS_MAX_RES:
                    for idx, image in enumerate(images):
                        if idx < num_images:
                            h, w = image.shape[-2:]
                            if h * w > CLS_MAX_RES * CLS_MAX_RES:
                                if random.random() < 0.99:
                                    loss_mask[idx] = 0
                                    # supervise 1% of the images with large resolution
                if ORIGIN_CLS_LOSS:
                    cls_loss = 1.0 * self.negative_cos_loss(cls_token, cls_token_teacher, loss_mask)
                else:
                    cls_loss = 10.0 * self.negative_cos_loss(cls_token, cls_token_teacher, loss_mask)
            else:
                if VQ_GEN_VIT:
                    raw_image_features, img_sizes, rec_loss, commit_loss_list, feat_rec_loss_list, ae_loss_list, kept_ratios, kept_ratios_first_code = self.get_model().get_vision_tower()(images)
                else:
                    raw_image_features, img_sizes = self.get_model().get_vision_tower()(images)




            image_features = [self.get_model().mm_projector(x, s).squeeze(0) for x, s in zip(raw_image_features, img_sizes)]
            
            assert len(image_features) == max_num_images, f"len(image_features)={len(image_features)} != max_num_images={max_num_images}"
            if VQ_GEN_VIT:
                assert len(commit_loss_list) == max_num_images, f"len(commit_loss_list)={len(commit_loss_list)} != max_num_images={max_num_images}"
                assert len(feat_rec_loss_list) == max_num_images, f"len(feat_rec_loss_list)={len(feat_rec_loss_list)} != max_num_images={max_num_images}"
                assert len(ae_loss_list) == max_num_images, f"len(ae_loss_list)={len(ae_loss_list)} != max_num_images={max_num_images}"
            
            if VQ_GEN_VIT:                
                loss_aux_mask = torch.zeros(len(commit_loss_list), dtype=commit_loss_list[0].dtype, device=commit_loss_list[0].device)
                loss_aux_mask[:num_images] = 1
            zero_feature = sum([x.mean() for x in image_features]) * 0
            if max_num_images > num_images:
                image_features = image_features[:num_images]

        else:
            assert TRAIN_CLS_TOKEN == False
            image_features = self.encode_images(images)
            image_features = [image_features[i] for i in range(image_features.shape[0])]
        
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            
            

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        _num_images_list = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            _num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            _num_images_list.append(_num_images)
            if _num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                if VQ_GEN_VIT:
                    loss_aux_mask[cur_image_idx] = 0
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(_num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < _num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        if cur_image_idx != num_images:
            print(f"{cur_image_idx}/{num_images}. For you information _num_images_list for each prompt {_num_images_list}. num_images = {num_images}. len(image_features) = {len(image_features)}. max_num_images for different device {max_num_images}. Length of input_ids {len(input_ids)}")
        if VQ_GEN_VIT:
            num_supervised = loss_aux_mask.sum()
            if num_supervised < 0.5:
                commit_loss = sum([ _loss * weight for _loss, weight in zip(commit_loss_list, loss_aux_mask)])
                feat_rec_loss = sum([ _loss * weight for _loss, weight in zip(feat_rec_loss_list, loss_aux_mask)])
                ae_loss = sum([ _loss * weight for _loss, weight in zip(ae_loss_list, loss_aux_mask)])
            else:
                commit_loss = sum([ _loss * weight for _loss, weight in zip(commit_loss_list, loss_aux_mask)]) / num_supervised
                feat_rec_loss = sum([ _loss * weight for _loss, weight in zip(feat_rec_loss_list, loss_aux_mask)]) / num_supervised
                ae_loss = sum([ _loss * weight for _loss, weight in zip(ae_loss_list, loss_aux_mask)]) / num_supervised

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0) + zero_feature
        
        # make ds happy
        if ZERO_IMAGE_FORWARD:
            zero_image = torch.zeros(1, 3, 28, 28).to(self.device).to(new_input_embeds.dtype)
            zero_img_feature = self.encode_images(zero_image).mean() * 0
            new_input_embeds = new_input_embeds + zero_img_feature



        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None


        if TRAIN_CLS_TOKEN:
            if VQ_GEN_VIT:
                return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, cls_loss, rec_loss, commit_loss, feat_rec_loss, ae_loss, kept_ratios, kept_ratios_first_code
            else:
                return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, cls_loss, None, None, None, None, None, None
        else:
            if VQ_GEN_VIT:
                return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, rec_loss, commit_loss, feat_rec_loss, ae_loss, kept_ratios, kept_ratios_first_code
            else:
                return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None, None, None, None, None, None, None

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
