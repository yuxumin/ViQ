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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.generation.utils import GenerateOutput

from llava_viq.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from dataclasses import dataclass
from copy import deepcopy


@dataclass
class CausalLMOutputWithPastWithCls(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    loss_cls: Optional[torch.FloatTensor] = None

    loss_rec: Optional[torch.FloatTensor] = None
    loss_commit: Optional[torch.FloatTensor] = None
    loss_feat_rec: Optional[torch.FloatTensor] = None
    loss_ae: Optional[torch.FloatTensor] = None
    codes_kept_ratios: Optional[torch.FloatTensor] = None
    codes_kept_ratios_first_code: Optional[torch.FloatTensor] = None
    token_metrics: Optional[dict] = None

class LlavaConfig(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2Model, self).__init__(config)

class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        self.model = LlavaQwen2Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.token_metrics = {
            "seen_tokens": 0,
            "seen_seq": 0,
            "seen_total_tokens": 0,
            'seen_images': 0,
        }

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_highres: Optional[List[torch.FloatTensor]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                cls_loss,
                rec_loss,
                commit_loss,
                feat_rec_loss,
                ae_loss,
                kept_ratios,
                kept_ratios_first_code
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_highres,
                image_sizes
            )

        if type(images) is list:
            self.token_metrics['seen_images'] += len(images)
        else:
            self.token_metrics['seen_images'] += images.size(0)

        llm_output = self.forward_llm_efficient(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cls_loss=cls_loss,
            rec_loss=rec_loss,
            commit_loss=commit_loss,
            feat_rec_loss=feat_rec_loss,
            ae_loss=ae_loss,
            kept_ratios=kept_ratios,
            kept_ratios_first_code=kept_ratios_first_code
        )


        return llm_output
    

    def forward_llm_efficient(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cls_loss, rec_loss, commit_loss, feat_rec_loss, ae_loss, kept_ratios, kept_ratios_first_code):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_dim = hidden_states.size(-1)
        shift_labels = labels[..., 1:].contiguous().reshape(-1)
        shift_hidden_states = hidden_states[..., :-1, :].contiguous().reshape(-1, hidden_dim)
        assert shift_labels.size(0) == shift_hidden_states.size(0)
        mask = shift_labels > -1

        seen_tokens = mask.float().sum().item()
        if not seen_tokens > 0:
            logits = self.lm_head(shift_hidden_states[0:2])
            loss = logits.sum() * 0
        else:
            shift_labels = shift_labels[mask]
            shift_hidden_states = shift_hidden_states[mask, :]
            logits = self.lm_head(shift_hidden_states)
            logits = logits.float()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, shift_labels)
        

        self.token_metrics['seen_tokens'] += int(seen_tokens)
        self.token_metrics['seen_seq'] += int(hidden_states.size(0))
        self.token_metrics['seen_total_tokens'] += int(hidden_states.size(0) * hidden_states.size(1))

        token_metrics = deepcopy(self.token_metrics)


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
    

        return CausalLMOutputWithPastWithCls(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_cls=cls_loss,
            loss_rec=rec_loss,
            loss_commit=commit_loss,
            loss_feat_rec=feat_rec_loss,
            loss_ae=ae_loss,
            codes_kept_ratios=kept_ratios,
            codes_kept_ratios_first_code=kept_ratios_first_code,
            token_metrics=token_metrics
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if image_aspect_ratio == 'anyres':
                images, highres_images = images[0::2], images[1::2]
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    highres_images,
                    image_sizes=image_sizes
                )
            else:
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    None,
                    image_sizes=image_sizes
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            eos_token_id=151645, # <|im_end|>
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_qwen2", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaQwen2ForCausalLM)
