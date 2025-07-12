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


class Llama(AbsLLM):
    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
        template_prompt: Optional[str] = None,
        dtype: str = "bfloat16",
        cache_dir: str = None,
        pad_token: str = "<unk>",
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ):
        super().__init__()

        assert model_name_or_path in [
            "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"
        ]
        self.is_llama2 = "Llama-2" in model_name_or_path

        logging.info(f"model_name_or_path: {model_name_or_path}")
        logging.info(f"dtype: {dtype}")
        logging.info(f"cache_dir: {cache_dir}")

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.template_prompt = template_prompt

        if template_prompt:
            # 先把整個 prompt tokenize 成 token list
            template_prompt_tokens = self.tokenizer.tokenize(template_prompt)
            # 我們要找到兩個占位符 ((HYP)) 和 ((BIAS)) 在 token list 裡的位置
            # Llama2 tokenize 會把 "((HYP))" 拆成長度 5 的 tokens，Llama3 則拆成長度 4
            len_hyp_indicator = 5 if self.is_llama2 else 4
            len_bias_indicator = 4  # Llama2 跟 Llama3 都是4

            # 先找第一個 ((HYP)) 的位置
            hyp_idx = None
            for i in range(len(template_prompt_tokens) - len_hyp_indicator + 1):
                if "".join(template_prompt_tokens[i : i + len_hyp_indicator]) == "((HYP))":
                    hyp_idx = i
                    break
            assert hyp_idx is not None, "在 template_prompt 找不到 ((HYP)) 標記"
    
            # 再找第一個 ((BIAS)) 的位置
            bias_idx = None
            for i in range(len(template_prompt_tokens) - len_bias_indicator + 1):
                if "".join(template_prompt_tokens[i : i + len_bias_indicator]) == "((BIAS))":
                    bias_idx = i
                    break
            assert bias_idx is not None, "在 template_prompt 找不到 ((BIAS)) 標記"

            # 確保先出現的是 ((HYP)) 再是 ((BIAS))（或者你可以允許任意順序，但要對應範例）
            if bias_idx < hyp_idx:
                raise ValueError("請先在 prompt 裡放 ((HYP))，再放 ((BIAS))")

            # 切割三段：
            #   prefix：從最前面到 ((HYP)) 前面
            #   middle：在 ((HYP)) 之後到 ((BIAS)) 前面
            #   suffix：在 ((BIAS)) 之後到結尾
            self.template_prefix_tokens = template_prompt_tokens[:hyp_idx]
            self.template_middle_tokens = template_prompt_tokens[hyp_idx + len_hyp_indicator : bias_idx]
            self.template_suffix_tokens = template_prompt_tokens[bias_idx + len_bias_indicator :]

            # 把三段都轉成 id list
            self.template_prefix_ids = (
                [self.lm.config.bos_token_id]  # Llama3: <|begin_of_text|>
                + self.tokenizer.convert_tokens_to_ids(self.template_prefix_tokens)
            )
            self.template_middle_ids = self.tokenizer.convert_tokens_to_ids(self.template_middle_tokens)
            self.template_suffix_ids = self.tokenizer.convert_tokens_to_ids(self.template_suffix_tokens)

            # 設置回應起始與結束 token
            if self.is_llama2:
                self.start_of_response_token_id = 29908  # "
                self.end_of_response_token_id = 29908    # "
            else:
                # Llama3
                self.start_of_response_token_id = 1  # "
                self.end_of_response_token_id = 1    # "

            logging.info(f"template_prompt: \n---\n{self.template_prompt}((RESPONSE))\n---")
            logging.info(f"template_prefix_ids: {self.template_prefix_ids}")
            logging.info(f"template_middle_ids: {self.template_middle_ids}")
            logging.info(f"template_suffix_ids: {self.template_suffix_ids}")
        else:
            # 如果沒有給 template，則照舊只設 start/end EOS
            self.start_of_response_token_id = self.lm.config.bos_token_id
            if self.is_llama2:
                self.end_of_response_token_id = self.lm.config.eos_token_id
            else:
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
        bias_in: List[str],
        bias_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 如果沒給 template，就照原本做
        if self.template_prompt is None:
            lm_in_pad, lm_in_lengths = res_in_pad, res_in_lengths
            lm_in_pad[lm_in_pad == -1] = self.pad_token_id
            return lm_in_pad, lm_in_lengths

        # 取出 __init__ 裡拆好的三段 id：prefix / middle / suffix
        prefix_ids = res_in_pad.new_tensor(self.template_prefix_ids, dtype=torch.long)
        middle_ids = res_in_pad.new_tensor(self.template_middle_ids, dtype=torch.long)
        suffix_ids = res_in_pad.new_tensor(self.template_suffix_ids, dtype=torch.long)

        # 1. 處理假設 hyp_in 和 bias_in 都是 List[str]
        if isinstance(hyp_in[0], str):
            lm_in = []
            lm_in_lengths = []
            for i, hyp in enumerate(hyp_in):
                # a) CTC hyp 轉 id
                hyp_ids = self.tokenizer(
                    hyp, return_tensors="pt"
                ).input_ids[0][1:].to(res_in_pad.device) # remove sos
                # b) bias 轉 id (若為空就給空 tensor)
                if bias_in[i] != "":
                    bias_ids = self.tokenizer(bias_in[i], return_tensors="pt").input_ids[0][1:].to(res_in_pad.device)
                else:
                    bias_ids = res_in_pad.new_tensor([], dtype=torch.long)

                # c) 真正的 decoder input (去掉 -1)
                res_part = res_in_pad[i][res_in_pad[i] != -1]

                # d) 按順序拼接 prefix + hyp + bias + suffix + res_part
                full_ids = torch.cat(
                    [
                        prefix_ids,    # 模板前綴
                        hyp_ids,       # CTC hyp
                        middle_ids,    # 模板中段
                        bias_ids,      # 偏向詞
                        suffix_ids,    # 模板後綴
                        res_part       # 真正要預測的 token
                    ],
                    dim=0,
                )
                lm_in.append(full_ids)

                # e) 長度 = 各段長度總和
                lm_in_lengths.append(
                    prefix_ids.size(0)
                    + hyp_ids.size(0)
                    + middle_ids.size(0)
                    + bias_ids.size(0)
                    + suffix_ids.size(0)
                    + res_in_lengths[i]
                )

            lm_in_pad = pad_list(lm_in, self.pad_token_id)
            # lm_in_lengths = torch.tensor(lm_in_lengths, device=res_in_pad.device, dtype=torch.long)
            lm_in_lengths = torch.stack(lm_in_lengths)

        # 2. 處理假設 hyp_in 是 List[Tensor]（id 序列），bias_in 仍視為 List[str]
        else:
            lm_in = []
            length_list = []
            for i, hyp_ids in enumerate(hyp_in):
                # 取出對應的 bias_ids (Tensor)：
                if bias_in[i] != "":
                    bias_ids_i = self.tokenizer(bias_in[i], return_tensors="pt").input_ids[0][1:].to(res_in_pad.device)
                else:
                    bias_ids_i = res_in_pad.new_tensor([], dtype=torch.long)
                # 真正要預測的 token：
                res_part = res_in_pad[i][res_in_pad[i] != -1]
                # 拼 full_ids_i：
                full_ids_i = torch.cat(
                    [
                        prefix_ids,
                        hyp_ids,
                        middle_ids,
                        bias_ids_i,
                        suffix_ids,
                        res_part
                    ], dim=0
                )
                lm_in.append(full_ids_i)
                length_list.append(full_ids_i.size(0))  # 直接用 full_ids_i 長度

            lm_in_pad = pad_list(lm_in, self.pad_token_id)
            lm_in_lengths = torch.tensor(length_list, device=res_in_pad.device, dtype=torch.long)

        return lm_in_pad, lm_in_lengths

    def forward(
        self,
        hyps_ctc: Union[List[torch.Tensor], List[str]],
        ctc_lengths: torch.Tensor,
        hyps_bias: List[str],
        bias_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hyps_ctc:       List[str]（或 List[Tensor]；以字串為例）── CTC 解碼後的文字
        ctc_lengths:    Tensor, shape=(B,) ── CTC 解碼後文字長度 placeholder（通常全 -1）
        hyps_bias:      List[str] ── 偏向詞列表（human‐readable）
        bias_lengths:   Tensor, shape=(B,) ── 偏向詞文本長度 placeholder（通常全 -1）
        res_in_pad:     Tensor, shape=(B, T_dec) ── decoder 輸入 token ID（已 pad）
        res_in_lengths: Tensor, shape=(B,)     ── decoder 輸入長度（包含 SOS/EOS）

        回傳 (lm_hidden, res_in_lengths)：其中
          - lm_hidden: Tensor, shape=(B, T_dec, H) ── 最後一層 LLM 的 hidden outputs，
                       並只保留「回應」這段時間步（用 res_in_lengths 截斷）。
          - res_in_lengths: 同輸入，給上層 ASR decoder 使用。
        """

        # 1. 把 CTC hyp 與 bias hyp、一併與 decoder 的 res_in_pad、res_in_lengths 傳到 prepare_prompt
        #    期待 prepare_prompt 回傳完整的 lm_in/input_ids 和對應的 lm_in_lengths。
        lm_in, lm_in_lengths = self.prepare_prompt(
            hyps_ctc,
            ctc_lengths,
            hyps_bias,
            bias_lengths,
            res_in_pad,
            res_in_lengths,
        )

        # 2. 由 lm_in_lengths 建立 attention_mask
        #    make_pad_mask(lm_in_lengths) → (B, max_len) 的 mask，1 表示 pad 位
        #    再取反 → 0 表示 pad，1 表示非 pad
        mask = (~make_pad_mask(lm_in_lengths)).to(lm_in.device).int()

        # 3. 把 lm_in 和 mask 丟給 HuggingFace Llama，要求回傳 hidden_states
        args = {
            "input_ids": lm_in,          # shape=(B, T_lm)
            "attention_mask": mask,      # shape=(B, T_lm)
            "use_cache": False,
            "output_hidden_states": True,
            "return_dict": True,
        }
        output = self.lm(**args).hidden_states[-1]
        # output 的 shape 為 (B, T_lm, H)

        # 4. 如果沒有 template_prompt，就直接把整段 hidden 當作回傳
        if self.template_prompt is None:
            # 直接回傳 LLM 的 hidden_states 最後一層，和原本的 res_in_lengths
            return output, res_in_lengths

        # 5. 如果有 template_prompt，就要把「完整輸入」(prefix+ctc+bias+suffix+res)
        #    當作一大段，中間只保留「真正要回應的那幾步」(res_in_lengths 長度)。
        #    假設 lm_in_lengths[i] = prefix + ctc + bias + suffix + res_in_len，
        #    真正回應部分就從 index (lm_in_lengths[i] - res_in_lengths[i]) 取到 lm_in_lengths[i]。
        B = output.size(0)
        ret = []
        for i in range(B):
            # o_i: Tensor, shape=(T_lm_i, H)
            o_i = output[i]  
            # 對第 i 筆 sample：
            #   lm_in_lengths[i]    ← prefix+ctc+bias+suffix+res_in_lengths[i]
            #   res_in_lengths[i]   ← 真正要保留的「回應」長度
            start_idx = lm_in_lengths[i] - res_in_lengths[i]
            end_idx = lm_in_lengths[i]
            ret.append(o_i[start_idx:end_idx])  # shape=(res_in_lengths[i], H)

        # 6. 因為不同 sample 的 res_in_lengths[i] 可能不同，要 pad 成同一長度
        lm_hidden = pad_list(ret, 0.0)  # shape=(B, max_res_len, H)

        return lm_hidden, res_in_lengths

    def prepare_prompt_for_inference(
        self,
        hyp_in: Union[List[torch.Tensor], List[str]],
        hyp_in_lengths: torch.Tensor,
        bias_in: List[str],
        bias_in_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = len(hyp_in)
        if self.template_prompt is None:
            lm_in, lm_in_lengths = res_in_pad, res_in_lengths
        else:
            prefix_ids = res_in_pad[0].new(
                self.template_prefix_ids
            ).repeat(len(hyp_in), 1)
            middle_ids = res_in_pad[0].new(
                self.template_middle_ids
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

            # bias 轉 id (若為空就給空 tensor)
            if bias_in[0] != "":
                # 先拿單筆的 id（shape [bias_len]）
                bias_id = self.tokenizer(
                    bias_in[0], return_tensors="pt"
                ).input_ids[0][1:].to(res_in_pad.device) # remove sos
                # 變成 (1, bias_len) 再 repeat → (B, bias_len)
                bias_ids = bias_id.unsqueeze(0).repeat(B, 1)
            else:
                # 先建一個空的一維 tensor，shape = (0,)
                empty = res_in_pad.new_tensor([], dtype=torch.long)  # shape (0,)
                # 再改成 (1, 0) 然後 repeat → (B, 0)
                bias_ids = empty.unsqueeze(0).repeat(B, 1)  # shape = (B, 0)

            lm_in = torch.cat(
                (prefix_ids, hyp_ids, middle_ids, bias_ids, suffix_ids, res_in_pad),
                dim=-1,
            )
            lm_in_lengths = (
                prefix_ids.size(1)
                + hyp_in_lengths
                + middle_ids.size(1)
                + bias_ids.size(1)
                + suffix_ids.size(1)
                + res_in_lengths
            )

        return lm_in, lm_in_lengths

    def forward_inference(
        self,
        hyps_ctc: Union[List[torch.Tensor], List[str]],
        ctc_lengths: torch.Tensor,
        hyps_bias: List[str],
        bias_lengths: torch.Tensor,
        res_in_pad: torch.Tensor,
        res_in_lengths: torch.Tensor,
        log_softmax: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(res_in_lengths == res_in_lengths[0])

        lm_in, lm_in_lengths = self.prepare_prompt_for_inference(
            hyps_ctc,
            ctc_lengths,
            hyps_bias,
            bias_lengths,
            res_in_pad,
            res_in_lengths,
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
            # "attention_mask": mask,
            "past_key_values": cache,
            "use_cache": True,
            "output_hidden_states": not log_softmax,
            "return_dict": True,
            "cache_position": cache_position,
        }

        output = self.lm(**args)

        # (32 ,2, (batch, head, len, dim))
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
