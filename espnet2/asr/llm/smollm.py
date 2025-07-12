#!/usr/bin/env python3

"""Hugging Face Transformers SmolLM."""
import logging
from typing import Any, List, Tuple, Optional, Union

import torch
from typeguard import typechecked

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, pad_list
from espnet2.asr.llm.abs_llm import AbsLLM

from transformers import AutoTokenizer, AutoModelForCausalLM


class SmolLM(AbsLLM):
    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
        template_prompt: Optional[str] = None,
        dtype: str = "bfloat16",
        cache_dir: str = None,
        pad_token: str = "<|endoftext|>",  # SmolLM uses <|endoftext|> as pad token
    ):
        super().__init__()

        # Updated to support SmolLM model variants
        assert model_name_or_path in [
            "HuggingFaceTB/SmolLM-135M",
            "HuggingFaceTB/SmolLM-360M", 
            "HuggingFaceTB/SmolLM-1.7B",
            "HuggingFaceTB/SmolLM-135M-Instruct",
            "HuggingFaceTB/SmolLM-360M-Instruct",
            "HuggingFaceTB/SmolLM-1.7B-Instruct",
            "HuggingFaceTB/SmolLM2-135M",
            "HuggingFaceTB/SmolLM2-360M",
            "HuggingFaceTB/SmolLM2-1.7B",
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        ]
        
        # Check if it's an instruct model
        self.is_instruct = "Instruct" in model_name_or_path
        self.is_smollm2 = "SmolLM2" in model_name_or_path

        logging.info(f"model_name_or_path: {model_name_or_path}")
        logging.info(f"dtype: {dtype}")
        logging.info(f"cache_dir: {cache_dir}")
        logging.info(f"is_instruct: {self.is_instruct}")
        logging.info(f"is_smollm2: {self.is_smollm2}")

        # Convert dtype string to torch dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir, 
            torch_dtype=torch_dtype,
            trust_remote_code=True  # May be needed for some SmolLM variants
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.template_prompt = template_prompt
        if template_prompt:
            assert "\"((HYP))\"" in template_prompt

            template_prompt_tokens = self.tokenizer.tokenize(template_prompt)
            len_hyp_indicator = 4  # SmolLM typically uses 4 tokens for ((HYP))
            
            for i in range(len(template_prompt_tokens)):
                if "".join(template_prompt_tokens[i: i + len_hyp_indicator]) == "((HYP))":
                    self.template_prefix_tokens = template_prompt_tokens[:i]
                    self.template_suffix_tokens = template_prompt_tokens[i + len_hyp_indicator:]
                    break

            self.template_prefix_ids = (
                [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
            ) + self.tokenizer.convert_tokens_to_ids(self.template_prefix_tokens)
            
            self.template_suffix_ids = self.tokenizer.convert_tokens_to_ids(self.template_suffix_tokens)

            # SmolLM token configuration
            if self.is_instruct:
                # For instruct models, might use special tokens
                self.start_of_response_token_id = self.tokenizer.bos_token_id or 1
                self.end_of_response_token_id = self.tokenizer.eos_token_id or 2
            else:
                # For base models
                self.start_of_response_token_id = self.tokenizer.bos_token_id or 1
                self.end_of_response_token_id = self.tokenizer.eos_token_id or 2

            logging.info(f"template_prompt: \n---\n{self.template_prompt}((RESPONSE))\n---")
            logging.info(f"template_prefix_ids: {self.template_prefix_ids}")
            logging.info(f"template_suffix_ids: {self.template_suffix_ids}")
        else:
            self.start_of_response_token_id = self.tokenizer.bos_token_id or 1
            self.end_of_response_token_id = self.tokenizer.eos_token_id or 2

        # Handle pad token
        if pad_token in self.tokenizer.vocab:
            self.pad_token_id = self.tokenizer.vocab[pad_token]
        else:
            self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        logging.info(f"start_of_response_token_id: {self.start_of_response_token_id}")
        logging.info(f"start_of_response_token: {self.tokenizer.convert_ids_to_tokens([self.start_of_response_token_id])}")
        logging.info(f"end_of_response_token_id: {self.end_of_response_token_id}")
        logging.info(f"end_of_response_token: {self.tokenizer.convert_ids_to_tokens([self.end_of_response_token_id])}")
        logging.info(f"pad_token_id: {self.pad_token_id}")
        logging.info(f"pad_token: {self.tokenizer.convert_ids_to_tokens([self.pad_token_id])}")

    def prepare_prompt(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
            lm_in_pad[lm_in_pad == -1] = self.pad_token_id
        else:
            prefix_ids = res_in_pad[0].new(self.template_prefix_ids)
            suffix_ids = res_in_pad[0].new(self.template_suffix_ids)

            if isinstance(hyp_in[0], str):
                lm_in = []
                lm_in_lengths = []
                for i, hyp in enumerate(hyp_in):
                    hyp_encoding = self.tokenizer(
                        hyp, return_tensors="pt", add_special_tokens=False
                    )
                    hyp_ids = hyp_encoding.input_ids[0].to(res_in_pad.device)
                    
                    lm_in.append(
                        torch.cat(
                            [
                                prefix_ids,
                                hyp_ids,
                                suffix_ids,
                                res_in_pad[i][res_in_pad[i] != -1]
                            ],
                            dim=0,
                        )
                    )
                    lm_in_lengths.append(
                        prefix_ids.size(0)
                        + hyp_ids.size(0)
                        + suffix_ids.size(0)
                        + res_in_lengths[i]
                    )
                lm_in_pad = pad_list(lm_in, self.pad_token_id)
                lm_in_lengths = torch.stack(lm_in_lengths)
            else:
                lm_in = [
                    torch.cat(
                        [
                            prefix_ids,
                            hyp,
                            suffix_ids,
                            res_in_pad[i][res_in_pad[i] != -1]
                        ],
                        dim=0,
                    ) for i, hyp in enumerate(hyp_in)
                ]
                lm_in_pad = pad_list(lm_in, self.pad_token_id)
                lm_in_lengths = (
                    prefix_ids.size(0)
                    + hyp_in_lengths
                    + suffix_ids.size(0)
                    + res_in_lengths
                )

        return lm_in_pad, lm_in_lengths

    def forward(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lm_in, lm_in_lengths = self.prepare_prompt(
            hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
        )
        mask = (~make_pad_mask(lm_in_lengths)).to(lm_in.device).int()

        args = {
            "input_ids": lm_in,
            "attention_mask": mask,
            "use_cache": False,
            "output_hidden_states": True,
            "return_dict": True,
        }

        output = self.lm(**args).hidden_states[-1]

        if self.template_prompt is None:
            return output, res_in_lengths
        else:
            ret = []
            for i, o in enumerate(output):
                ret.append(o[lm_in_lengths[i] - res_in_lengths[i]: lm_in_lengths[i]])

            return pad_list(ret, 0.0), res_in_lengths

    def prepare_prompt_for_inference(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
        else:
            prefix_ids = res_in_pad[0].new(
                self.template_prefix_ids
            ).repeat(len(hyp_in), 1)
            suffix_ids = res_in_pad[0].new(
                self.template_suffix_ids
            ).repeat(len(hyp_in), 1)

            if isinstance(hyp_in[0], str):
                hyp_encoding = self.tokenizer(
                    hyp_in[0], return_tensors="pt", add_special_tokens=False
                )
                hyp_id = hyp_encoding.input_ids[0].to(res_in_pad.device)
                hyp_ids = hyp_id.repeat(len(hyp_in), 1)
            else:
                hyp_ids = torch.stack(hyp_in)

            lm_in = torch.cat(
                (prefix_ids, hyp_ids, suffix_ids, res_in_pad),
                dim=-1,
            )
            lm_in_lengths = (
                prefix_ids.size(-1)
                + hyp_ids.size(-1)
                + suffix_ids.size(-1)
                + res_in_lengths
            )

        return lm_in, lm_in_lengths

    def forward_inference(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
        log_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(res_in_lengths == res_in_lengths[0])

        lm_in, lm_in_lengths = self.prepare_prompt_for_inference(
            hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
        )
        mask = (~make_pad_mask(lm_in_lengths)).to(lm_in.device).int()

        args = {
            "input_ids": lm_in,
            "attention_mask": mask,
            "use_cache": False,
            "output_hidden_states": not log_softmax,
            "return_dict": True,
        }

        output = self.lm(**args)

        if log_softmax:
            output = torch.log_softmax(output.logits, dim=-1)
        else:
            output = output.hidden_states[-1]

        if self.template_prompt is None:
            return output, res_in_lengths
        else:
            return output[:, -res_in_lengths[0]:], res_in_lengths

    def forward_inference_cached(
        self,
        hyp_in: Union[List[torch.Tensor], List[str], None],
        hyp_in_lengths: Union[torch.Tensor, None],
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
        log_softmax: bool = False,
        cache: List[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        assert torch.all(res_in_lengths == res_in_lengths[0])

        if cache is None:
            lm_in, lm_in_lengths = self.prepare_prompt_for_inference(
                hyp_in, hyp_in_lengths, res_in_pad, res_in_lengths
            )
            cache_position = torch.arange(
                lm_in.shape[1]
            ).long().to(lm_in.device)
        else:
            lm_in = res_in_pad[:, -1].unsqueeze(-1)

            new_cache = []
            for layer, x in enumerate(cache):
                new_cache.append(
                    (
                        torch.cat(
                            [
                                self.prefix_cache[layer][0].repeat(x[0].shape[0], 1, 1, 1),
                                x[0]
                            ],
                            dim=2
                        ),
                        torch.cat(
                            [
                                self.prefix_cache[layer][1].repeat(x[1].shape[0], 1, 1, 1),
                                x[1]
                            ],
                            dim=2
                        )
                    )
                )
            cache = new_cache

            cache_position = torch.Tensor(
                [cache[0][0].shape[-2]]
            ).long().to(lm_in.device)

        args = {
            "input_ids": lm_in,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": not log_softmax,
            "return_dict": True,
        }
        
        # Add cache_position only if the model supports it
        try:
            output = self.lm(**args, cache_position=cache_position)
        except TypeError:
            # Fallback for models that don't support cache_position
            del args["cache_position"] if "cache_position" in args else None
            output = self.lm(**args)

        past_key_values = output.past_key_values
        if cache is None:
            self.prefix_cache = past_key_values
            new_past_key_values = []
            for x in past_key_values:
                batch_size, head_size, _, hdim = x[0].shape
                new_past_key_values.append(
                    (
                        torch.empty((batch_size, head_size, 0, hdim)).to(lm_in.device),
                        torch.empty((batch_size, head_size, 0, hdim)).to(lm_in.device)
                    )
                )
            past_key_values = new_past_key_values
        else:
            prefix_len = self.prefix_cache[0][0].shape[2]
            new_past_key_values = []
            for x in past_key_values:
                new_past_key_values.append(
                    (x[0][:, :, prefix_len:], x[1][:, :, prefix_len:])
                )
            past_key_values = new_past_key_values

        if log_softmax:
            output = torch.log_softmax(output.logits, dim=-1)
        else:
            output = output.hidden_states[-1]

        return output[:, -1].unsqueeze(1), past_key_values

    def output_size(self) -> int:
        """Get the output size."""
        return self.lm.config.hidden_size