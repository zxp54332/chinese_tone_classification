import torch
from transformers import Seq2SeqTrainer


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        # default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )

        if "decoder_input_ids" in inputs:
            index102 = (inputs["decoder_input_ids"] == 102).nonzero(as_tuple=True)[1]
            bsz = inputs["decoder_input_ids"].shape[0]
            index102 = index102[::bsz]
            if bsz == 1:
                index102 = index102[:1]
            tensors = []
            batch_max_length = 0
            for idx, tensor in zip(index102, inputs["decoder_input_ids"]):
                idx += 1
                tensors.append(tensor[:idx])
                if tensors[-1].shape[0] > batch_max_length:
                    batch_max_length = tensors[-1].shape[0]

            import torch.nn.functional as F

            for idx, tensor in enumerate(tensors):
                if tensor.shape[0] < batch_max_length:
                    tensors[idx] = F.pad(
                        tensor, (batch_max_length - tensor.shape[0], 0)
                    )
            decoder_input_ids = torch.stack(tensors, dim=0)
            gen_kwargs["decoder_input_ids"] = decoder_input_ids

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        gen_kwargs["max_new_tokens"] = 1
        gen_kwargs.pop("max_length")

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if (
            gen_kwargs.get("max_length") is not None
            and generated_tokens.shape[-1] < gen_kwargs["max_length"]
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[
            -1
        ] < (gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_new_tokens"] + 1
            )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if (
                gen_kwargs.get("max_length") is not None
                and labels.shape[-1] < gen_kwargs["max_length"]
            ):
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, (gen_kwargs["max_new_tokens"] + 1)
                )
        else:
            labels = None

        return (loss, generated_tokens, labels)
