#!/usr/bin/env python3

"""Hugging Face Transformers PLM."""
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


class Qwen(AbsLLM):
    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
        template_prompt: Optional[str] = None,
        dtype: str = "bfloat16",
        cache_dir: str = None,
        pad_token: str = "<unk>",
    ):
        super().__init__()

        assert model_name_or_path.startswith("Qwen/"), (
            f"Expected a Qwen model path, got {model_name_or_path}"
        )

        logging.info(f"model_name_or_path: {model_name_or_path}")
        logging.info(f"dtype: {dtype}")
        logging.info(f"cache_dir: {cache_dir}")

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, torch_dtype=dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.template_prompt = template_prompt
        if template_prompt:
            assert "\"((HYP))\"" in template_prompt
            # 把 tokens[i:i+len_hyp_indicator] 串起來，對比字串 "((HYP))"
            template_prompt_tokens = self.tokenizer.tokenize(template_prompt)
            len_hyp_indicator = 4  # Qwen2
            for i in range(len(template_prompt_tokens)):
                if "".join(template_prompt_tokens[i: i + len_hyp_indicator]) == "((HYP))":
                    self.template_prefix_tokens = template_prompt_tokens[:i]
                    self.template_suffix_tokens = template_prompt_tokens[i + len_hyp_indicator:]
                    break
            # 轉成 token id 序列
            self.template_prefix_ids = (
                [self.lm.config.bos_token_id]
                + self.tokenizer.convert_tokens_to_ids(self.template_prefix_tokens)
            )
            self.template_suffix_ids = self.tokenizer.convert_tokens_to_ids(self.template_suffix_tokens)

            # qwen2
            self.start_of_response_token_id = 1 # "
            self.end_of_response_token_id = 1 # "

            logging.info(f"template_prompt: \n---\n{self.template_prompt}((RESPONSE))\n---")
            logging.info(f"template_prefix_ids: {self.template_prefix_ids}")
            logging.info(f"template_suffix_ids: {self.template_suffix_ids}")
        else:
            self.start_of_response_token_id = self.lm.config.bos_token_id

            # didn't carefully check
            self.end_of_response_token_id = self.lm.config.eos_token_id[0]

        self.pad_token_id = self.tokenizer.vocab[pad_token]

        logging.info(f"start_of_response_token_id: {self.start_of_response_token_id}")
        logging.info(f"start_of_response_token: {self.tokenizer.convert_ids_to_tokens(self.start_of_response_token_id)}")
        logging.info(f"end_of_response_token_id: {self.end_of_response_token_id}")
        logging.info(f"end_of_response_token: {self.tokenizer.convert_ids_to_tokens(self.end_of_response_token_id)}")
        logging.info(f"pad_token_id: {self.pad_token_id}")
        logging.info(f"pad_token: {self.tokenizer.convert_ids_to_tokens(self.pad_token_id)}")

    def prepare_prompt(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
            # 把 -1（未填值）都換成 pad_token_id
            lm_in_pad[lm_in_pad == -1] = self.pad_token_id
        else:
            prefix_ids = res_in_pad[0].new(self.template_prefix_ids)
            suffix_ids = res_in_pad[0].new(self.template_suffix_ids)
    
            # List[str]
            if isinstance(hyp_in[0], str):
                lm_in = []
                lm_in_lengths = []
                for i, hyp in enumerate(hyp_in):
                    hyp_ids = self.tokenizer(
                        hyp, return_tensors="pt"
                    ).input_ids[0][1:].to(res_in_pad.device) # remove sos
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
            # List[torch.Tensor]
            else: 
                lm_in = [
                    torch.cat(
                        [
                            prefix_ids,                                       # 模板前綴的 token id 序列
                            hyp,                                              # 這條 sample 的 hypothesis Tensor
                            suffix_ids,                                       # 模板後綴的 token id 序列
                            res_in_pad[i][res_in_pad[i] != -1]                # 真正要預測的部分（去掉 pad=-1 的位置）
                        ],
                        dim=0,
                    ) for i, hyp in enumerate(hyp_in)
                ]
                lm_in_pad = pad_list(lm_in, self.pad_token_id)
                lm_in_lengths = (
                    prefix_ids.size(0)        # 前綴長度
                    + hyp_in_lengths          # 每筆 hypothesis 的長度（Tensor）
                    + suffix_ids.size(0)      # 後綴長度
                    + res_in_lengths          # 每筆實際要預測部分的長度（Tensor）
                )

        return lm_in_pad, lm_in_lengths

    def forward(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],   # ctc output
        hyp_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,   # decoder input
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
            for i, o in enumerate(output):  # o 的 shape: [T_i, H]
                # lm_in_lengths[i]       ← 整段輸入 (prefix+hyp+suffix+res) 的長度
                # res_in_lengths[i]     ← 真正回應 (res) 的長度
                # 所以起始位置 = lm_in_lengths[i] - res_in_lengths[i]
                # 只保留最後 res_in_lengths[i] 個時間步的 hidden state
                ret.append(o[lm_in_lengths[i] - res_in_lengths[i]: lm_in_lengths[i]])   # shape: [res_in_lengths[i], H]
            # 把各 sample 長度可能不同的 ret（list of [L_i, H]）pad 到同一最大長度 R_max
            # pad_list(..., 0.0) 會用 0.0 填充缺失的時間步
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
                hyp_id = self.tokenizer(
                    hyp_in[0], return_tensors="pt"
                ).input_ids[0][1:].to(res_in_pad.device) # remove sos
                hyp_ids = hyp_id.repeat(len(hyp_in), 1)
            else:
                hyp_ids = torch.stack(hyp_in)

            lm_in = torch.cat(
                (prefix_ids, hyp_ids, suffix_ids, res_in_pad),
                dim=-1,
            )
            lm_in_lengths = (
                prefix_ids.size(0)
                + hyp_in_lengths
                + suffix_ids.size(0)
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

    def output_size(self) -> int:
        """Get the output size."""
        return self.lm.config.hidden_size
