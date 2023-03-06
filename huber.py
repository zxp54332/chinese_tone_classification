from transformers import HubertModel, HubertForSequenceClassification
from baseline_models import TextClassificationModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from weight import weight_balance
from colored_text import bcolors

_HIDDEN_STATES_START_POSITION = 1

class HubertClassification(HubertForSequenceClassification):
    def __init__(self, config, train_data, vocab_size, text_embed_dim, with_text, num_class=5):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Hubert adapters (config.add_adapter=True)"
            )
        self.hubert = HubertModel(config)
        self.text_model = TextClassificationModel(
            vocab_size=vocab_size, embed_dim=text_embed_dim, num_class=num_class
        )
        self.text_norm = nn.LayerNorm(num_class)
        self.logits_norm = nn.LayerNorm(num_class)
        self.cat_norm = nn.LayerNorm(num_class * 2)
        self.with_text = with_text
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.fc = nn.Linear(10, 5)
        print(f"{bcolors.red}with_text：{with_text}")
        print(f"config.num_labels：{config.num_labels}{bcolors.reset}")

        # Initialize weights and apply final processing
        self.post_init()
        self.train_data = train_data
        self.weights = None
        if train_data is not None:
            self.weights = weight_balance(self.train_data)
            #self.register_buffer('weights', weight_balance(self.train_data))
            #self.weights = torch.tensor([1.0023, 0.8404, 1.2754, 0.6236, 2.3539],dtype=torch.float)


    def forward(
        self,
        input_values: Optional[torch.Tensor],
        ids: Optional[torch.Tensor] = None,
        max_pinyin_length: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)
        #logits = F.dropout(self.logits_norm(logits), training=self.training)
        if self.weights is not None:
            weights = self.weights.to(labels.device)

        loss = None
        if self.with_text:
            ids = torch.flatten(ids)
            offsets = torch.tensor(list(range(0, len(ids), max_pinyin_length[0])))
            offsets = offsets.to("cuda")
            text_logits = self.text_model(ids, offsets=offsets)
            text_logits = F.dropout(self.text_norm(text_logits), training=self.training)
            #text_logits = self.text_norm(text_logits)
            #cat_logits = self.fc(torch.cat((text_logits, logits), dim=-1))
            cat_logits = self.fc(F.dropout(self.cat_norm(torch.cat((text_logits, logits), dim=-1)), training=self.training))

            if labels is not None:
                if self.weights is not None:
                    loss_fct = CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(cat_logits.view(-1, self.config.num_labels), labels.view(-1))

            if not return_dict:
                output = (cat_logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=cat_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            if labels is not None:
                if self.weights is not None:
                    loss_fct = CrossEntropyLoss(weight=weights)
                else:
                    loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

            if not return_dict:
                output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )