from __future__ import annotations

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2CacheBase,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
import torch
import random
import threading
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook, ExLlamaV2PostSamplingResult

class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2CacheBase
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.Tensor | None

    abort_event: threading.Event | None


    def __init__(self,
                 model: ExLlamaV2,
                 cache: ExLlamaV2CacheBase,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.sequence_ids = None
        self.abort_event = None

    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 2), dtype = torch.long)
        self.model.forward(input_ids, cache = None, input_mask = None, preprocess_only = True)


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len


    def generate_simple(self,
                        prompt: str or list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        input_embedding: torch.Tensor = None,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event or None = None):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """

        if prompt is not None and input_embedding is not None:
            raise ValueError("Cannot provide both prompt and input_embedding")

        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Accept LoRA or list of LoRAs

        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]

        # Apply seed

        if seed is not None: random.seed(seed)
        unhealed_token = None
        # Tokenize input and produce padding mask if needed
        if prompt is not None:
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            ids, position_offsets = self.tokenizer.encode(prompt,
                                                          encode_special_tokens = encode_special_tokens,
                                                          return_offsets = True,
                                                          add_bos = add_bos)
            input_embedding = None
            overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
            if overflow > 0: ids = ids[:, overflow:]
            mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
            # Prepare for healing

            if ids.shape[-1] < 2: token_healing = False
            if token_healing:
                unhealed_token = ids[:, -1:]
                ids = ids[:, :-1]
        else:
            batch_size = input_embedding.shape[0]
            mask = None
            ids = None
            position_offsets = None

        if batch_size == 1: position_offsets = None

        # Process prompt and begin gen

        if prompt is not None: self._gen_begin_base(ids, None, mask, loras, position_offsets = position_offsets)
        if self.abort_event and self.abort_event.is_set():
            if isinstance(prompt, str): return ""
            else: return [""] * len(prompt)

        # Begin filters
        if prompt is not None:
            id_to_piece = self.tokenizer.get_id_to_piece_list()
            if unhealed_token is not None:
                unhealed_token_list = unhealed_token.flatten().tolist()
                heal = [id_to_piece[x] for x in unhealed_token_list]
            else:
                heal = None
        else:
            heal = None
        gen_settings.begin_filters(heal)

        # Generate tokens

        batch_eos = [False] * batch_size
        initial_fwd = True
        prob_list = []

        for i in range(num_tokens):

            if self.abort_event and self.abort_event.is_set():
                break
            if not initial_fwd: input_embedding = None  # release vram

            logits = self.model.forward(self.sequence_ids[:, -1:] if self.sequence_ids is not None else None,
                                        input_embedding if input_embedding is not None else None,
                                        self.cache,
                                        input_mask = mask,
                                        loras = loras,
                                        position_offsets = position_offsets).float().cpu()
            logits = logits[:, -1:, :]
            initial_fwd = False
            if self.sequence_ids is None:
                self.sequence_ids = torch.zeros((batch_size, 0), dtype = torch.long)  # same as embedding layer
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random(), self.tokenizer, prefix_token = unhealed_token)
            prob_list.append(prob)
            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if eos: break

        # Decode

        text = self.tokenizer.decode(self.sequence_ids, decode_special_tokens = decode_special_tokens)

        if isinstance(prompt, str): return text[0], prob_list
        return text, prob_list


    def _gen_begin_base(self,
                        input_ids: torch.Tensor,
                        input_embedding: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1] if input_ids is not None else None,
                           input_embedding,
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]
