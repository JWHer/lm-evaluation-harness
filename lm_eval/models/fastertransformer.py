import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from tqdm import tqdm
from typing import Any, List, Union, Optional

try:
    torch.classes.load_library(os.environ.get("FT_PATH"))
except Exception:
    raise ImportError(
        "Please install FasterTransformer and provide a path to the binary"
        "`libth_transformer.so` via the environment variable `FT_PATH`."
    )
from lm_eval.base import BaseLM
from lm_eval import utils


class FTLM(BaseLM):

    BOS_TOKEN = 0
    PAD_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="ft/opt-6.7b",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "fp32",
    ):
        super().__init__()
        # Additional args for faster transformer
        CKPT_DIR    = os.getenv('CKPT_DIR', '.')
        vocab_file  = os.path.join(CKPT_DIR, 'gpt2-vocab.json')
        vocab_size  = 50272
        merges_file = os.path.join(CKPT_DIR, 'gpt2-merges.txt')
        weight_path = os.path.join(CKPT_DIR, pretrained)
        int8_mode   = 0
        int8_weights, int8_scales = [], []
        predefined = {
            'ft/opt-6.7b': {
                'num_layers': 32,
                'num_heads' : 32,
                'embed_size': 2560,
            },
            'ft/opt-125m': {
                'num_layers': 12,
                'num_heads' : 12,
                'embed_size': 768,
            },
        }

        # Codes start here
        model, tokenizer, device = None, None, None
        dist.init_process_group(backend="mpi")
        world_size = dist.get_world_size()
        rank = dist.get_rank() % world_size

        # Initialize device
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
        self._device = device
        torch.cuda.set_device(device)
        print(f"Using device '{device}'")
        # revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
        # hellaswag compatible
        setattr(self.tokenizer, 'vocab_size', vocab_size)

        torch_dtypes = {"fp16": torch.half, "bf16": torch.bfloat16, "fp32": torch.float}
        dtype = torch_dtypes[dtype]

        state_dict = torch.load(f"{weight_path}/part-{rank}.pt")
        weights = [w.to(device, dtype) for w in state_dict["weights"]]
        
        # Initialize new model and tokenizer instances
        kwargs = {
            "head_num"      : predefined[pretrained]['num_heads'],
            "size_per_head" : predefined[pretrained]['embed_size'] // predefined[pretrained]['num_heads'],
            "inter_size"    : 4 * predefined[pretrained]['embed_size'],
            "layer_num"     : predefined[pretrained]['num_layers'],
            "expert_num"    : 0,
            "moe_k"         : 0,
            "moe_layer_index": [],
            "vocab_size"    : vocab_size,
            "start_id"      : 2,
            "end_id"        : 2,
            "tensor_para_size": world_size,
            "pipeline_para_size": 1,
            "int8_mode"     : int8_mode,
            "layernorm_eps": 1e-5,
            "layernorm_type": "pre_layernorm",
            "activation_type": "Relu",
            "has_positional_encoding": True,
            "has_pre_decoder_layernorm": False,
            "has_post_decoder_layernorm": True,
            "has_adapters": False,
            "adapter_inter_size": 0,
            "use_attention_linear_bias": False,
            "weights"       : weights,
            "int8_weights"  : int8_weights,
            "scale"         : int8_scales,
            "shared_contexts_ratio": 1.0,
        }
        self.model = torch.classes.FasterTransformer.ParallelGptOp(*kwargs.values())

        # self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        # FIXME: Same as output length?
        self._max_length = 128 # max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        # return self.tokenizer.eos_token_id
        return FTLM.EOS_TOKEN

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens.tolist())

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """

        # Faster Transformer args
        output_length = 128 # 256
        beam_width = 1
        top_k = 20
        top_p = 0.95
        diversity_rate = 0.0
        temperature = 0.7
        len_penalty = 0.0
        repetition_penalty = 1.2

        # object = [None]
        # if torch.distributed.get_rank() == 0:
        #     object = [[self.tokenizer.encode(inps).ids]]

        dist.broadcast_object_list(inps, src=0)
        output, cum_log_probs = self.generate(
            inps,
            output_length = output_length,
            beam_width    = beam_width,
            top_k         = top_k,
            top_p         = top_p,
            diversity_rate= diversity_rate,
            temperature   = temperature,
            len_penalty   = len_penalty,
            repetition_penalty= repetition_penalty,
            random_seed   = 0,
            return_cum_log_probs = 1,
        )
        # if torch.distributed.get_rank() == 0:
            # print(f"Output: {output[0][0]['text']}")
        return output, cum_log_probs

    @torch.inference_mode()
    def generate(
        self,
        inputs: List[List[int]],
        output_length: int,
        beam_width: int = 1,
        top_k: Optional[int] = 0,
        top_p: Optional[float] = 1.0,
        diversity_rate: Optional[float] = None,
        temperature: Optional[float] = 1.0,
        len_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        presence_penalty: Optional[float] = None,
        random_seed: Optional[int] = 0,
        min_length: Optional[int] = None,
        bad_words_list: Optional[torch.Tensor] = None,
        return_cum_log_probs: Optional[int] = 0,
    ) -> List[Any]:
        # inputs = [[FTLM.EOS_TOKEN] + toks for toks in inputs]
        inputs = [torch.tensor(toks, dtype=torch.int32, device=self.device) for toks in inputs]
        # FIXME: inputs = inputs.unsqueeze(0).type(torch.int32)
        lengths = torch.tensor([len(t) for t in inputs], dtype=torch.int32, device=self.device)
        inputs = nn.utils.rnn.pad_sequence(inputs, True, padding_value=FTLM.PAD_TOKEN)

        if top_k is not None:
            top_k = torch.tensor([top_k], dtype=torch.int32)
        if top_p is not None:
            top_p = torch.tensor([top_p], dtype=torch.float32)
        if diversity_rate is not None:
            diversity_rate = torch.tensor([diversity_rate], dtype=torch.float32)
        if temperature is not None:
            temperature = torch.tensor([temperature], dtype=torch.float32)
        if len_penalty is not None:
            len_penalty = torch.tensor([len_penalty], dtype=torch.float32)
        if repetition_penalty is not None:
            repetition_penalty = torch.tensor([repetition_penalty], dtype=torch.float32)
        if presence_penalty is not None:
            presence_penalty = torch.tensor([presence_penalty], dtype=torch.float32)
        if random_seed is not None:
            random_seed = torch.tensor([random_seed], dtype=torch.int64)
        if min_length is not None:
            min_length = torch.tensor([min_length], dtype=torch.int64)

        outputs, output_lengths, cum_log_probs = self.model.forward(
            inputs,
            lengths,
            output_length,
            beam_width,
            top_k,
            top_p,
            diversity_rate,
            temperature,
            len_penalty,
            repetition_penalty,
            presence_penalty,
            min_length,
            random_seed,
            bad_words_list,
            return_cum_log_probs,
        )

        results = []
        beam_idx = 0
        special = outputs.new_tensor([FTLM.BOS_TOKEN, FTLM.PAD_TOKEN, FTLM.EOS_TOKEN, FTLM.UNK_TOKEN])
        for output, output_len in zip(outputs, output_lengths):
            mask = ~torch.isin(output[beam_idx], special)
            mask[1:] = mask[1:].cummin(dim=0)[0]

            tokens = output[beam_idx][1 : output_len[beam_idx]]
            tokens = tokens[mask[1 : output_len[beam_idx]]]
            results.append(tokens)
            # results.append({"text": self.tokenizer.decode(tokens.tolist())})
        return results, cum_log_probs

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in utils.chunks(
            tqdm(reordered_requests, disable=disable_tqdm),
            n=self.batch_size
            if self.batch_size != "auto"
            else override_bs
            if override_bs is not None
            else 0,
            fn=_batch_scheduler
            if self.batch_size == "auto"
            and n_reordered_requests > 0
            and not override_bs
            else None,
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = (
                    padding_length if padding_length is not None else inplen
                )

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            greedy_tokens_list, logits_sum_list = self._model_call(batched_inps) # [batch, padding_length, vocab]

            for (cache_key, _, _), greedy_tokens, logits_sum, inp, inplen, cont_toks in zip(
                chunk, greedy_tokens_list, logits_sum_list, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                # inplen = inplen + (
                #     logits.shape[0] - padding_length
                # )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                # logits = logits[inplen - contlen : inplen].unsqueeze(
                #     0
                # )  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                # greedy_tokens = logits.argmax(dim=-1)
                greedy_tokens = greedy_tokens[inplen - contlen : inplen].cpu()
                cont_toks = torch.tensor(cont_toks, dtype=torch.int32) # [1, seq]
                if greedy_tokens.shape != cont_toks.shape:
                    print(f'Size mismatch! g:{greedy_tokens.shape} c:{cont_toks.shape}')
                    max_equal = False
                else:
                    max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                # logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                #     -1
                # )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (logits_sum.cpu(), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)
