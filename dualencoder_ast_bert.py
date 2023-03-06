from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, Trainer
from transformers.models.clip.modeling_clip import CLIPOutput
from torch import nn
import torch
import copy


class VisionTextDualEncoderConfig(PretrainedConfig):
    model_type = "vision-text-dual-encoder"
    is_composition = True

    def __init__(
        self,
        vision_model_name,
        text_model_name,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = AutoConfig.from_pretrained(vision_model_name, max_length=200) # 2.0s
        self.text_config = AutoConfig.from_pretrained(text_model_name)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_configs(cls, vision_config, text_config, **kwargs):
        return cls(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


# Copied from transformers.models.clip.modeling_clip.clip_loss
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class AudioTextDualEncoderModel(PreTrainedModel):
    def __init__(self, config: VisionTextDualEncoderConfig, vision_model, text_model):

        super().__init__(config)

        self.vision_model = vision_model
        self.text_model = text_model

        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.ones([]) * self.config.logit_scale_init_value
        )

    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            # token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #  print("text_outputs", text_outputs.last_hidden_state[0, 0, 0]) # tensor(0.8026)

        pooled_output = text_outputs.last_hidden_state[:, 0]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        input_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        vision_outputs = self.vision_model(
            input_values=input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs.pooler_output  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids=None,
        input_values=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        labels=None,
        return_loss=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        return_dict = True

        vision_outputs = self.vision_model(
            input_values=input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs.pooler_output # pooler_output
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs.last_hidden_state[:, 0]  # pooler_output
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T

        loss = clip_loss(logits_per_text) if return_loss else None
        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return (outputs.loss, outputs) if return_outputs else outputs.loss
