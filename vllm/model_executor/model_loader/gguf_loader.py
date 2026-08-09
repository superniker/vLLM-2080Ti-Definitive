# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Generator
from typing import TYPE_CHECKING, cast

import gguf
import regex as re
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from vllm import _custom_ops as ops
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf,
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
    gguf_quant_weights_iterator_multi,
)
from vllm.transformers_utils.gguf_utils import detect_gguf_multimodal
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        # repo_id:quant_type
        elif "/" in model_name_or_path and ":" in model_name_or_path:
            repo_id, quant_type = model_name_or_path.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        """Discover all GGUF shard files from a single shard path.

        Supports variable-width shard indices by dynamically detecting
        the padding from the original filename.
        E.g. ``*-00001-of-00005.gguf`` → all 5 shards,
             ``*-01-of-15.gguf`` → all 15 shards.
        """
        match = re.search(r"-(\d+)-of-(\d+)\.gguf$", model_path)
        if not match:
            return [model_path]
        total = int(match.group(2))
        num_digits = len(match.group(1))
        prefix = model_path[: match.start(1)]
        suffix = model_path[match.end(2) :]
        files = []
        for i in range(1, total + 1):
            shard_path = f"{prefix}{i:0{num_digits}d}-of-{total:0{num_digits}d}{suffix}"
            if os.path.isfile(shard_path):
                files.append(shard_path)
        if files:
            logger.info("Discovered %d GGUF shard files", len(files))
        return files if files else [model_path]

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        config = model_config.hf_config
        # Get text config to handle both nested (multimodal) and flat
        # (text-only) config structures. For multimodal models like
        # Gemma3Config, this returns config.text_config. For text-only
        # models, this returns config itself.
        text_config = config.get_text_config()
        model_type = config.model_type
        # Qwen3_5Config 总是创建 vision_config,需以 mmproj 文件为准
        is_multimodal = detect_gguf_multimodal(model_config.model) is not None
        gguf_to_hf_name_map = {}
        sideload_params: list[re.Pattern] = []
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            # GGUF layer map assumes merged expert weights
            # map them manually like deepseek2
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        text_num_layers = text_config.num_hidden_layers
        text_name_map = gguf.get_tensor_name_map(arch, text_num_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_num_layers = getattr(
                config.vision_config, "num_hidden_layers", None
            ) or getattr(config.vision_config, "depth", None)
            vision_name_map = gguf.get_tensor_name_map(mm_proj_arch, vision_num_layers)
        else:
            vision_name_map = None

        # Create dummy model to extract parameter names
        # For multimodal: use AutoModelForImageTextToText to get
        # language + vision + projector params
        # For text-only: use AutoModelForCausalLM to get language model params
        auto_cls = (
            AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        )
        with torch.device("meta"):
            try:
                dummy_model = auto_cls.from_config(
                    config, trust_remote_code=model_config.trust_remote_code
                )
            except ValueError:
                # Qwen3_5Config 未注册到 transformers AutoModel(GGUF qwen35 场景)。
                # vLLM Qwen3_5 继承 Qwen3Next(混合注意力),用 Qwen3NextForCausalLM
                # 对 text_config 生成同构参数名。
                logger.info(
                    "AutoModel 无法识别 %s,改用 Qwen3NextForCausalLM 生成参数名映射",
                    type(config).__name__,
                )
                from transformers.models.qwen3_next.configuration_qwen3_next import (
                    Qwen3NextConfig,
                )
                from transformers.models.qwen3_next.modeling_qwen3_next import (
                    Qwen3NextForCausalLM,
                )

                tc = config.get_text_config()
                q3n = Qwen3NextConfig(
                    **{
                        k: v
                        for k, v in vars(tc).items()
                        if not k.startswith("_") and k != "model_type"
                    },
                    # Qwen3.5-27B 是稠密模型:关掉 Qwen3Next 默认的 MoE 分支
                    num_experts=1,
                    num_experts_per_tok=1,
                    mlp_only_layers=[],
                    decoder_sparse_step=10**9,  # 全层 dense MLP
                )
                dummy_model = Qwen3NextForCausalLM(q3n)

        state_dict = dummy_model.state_dict()
        # Qwen3.5 GGUF:ssm_alpha/ssm_beta 分开存储,vLLM 的 Qwen3_5 load_weights
        # 按 in_proj_b/shard0 + in_proj_a/shard1 分片加载,把合并名拆开
        if model_type == "qwen35":
            renamed: dict = {}
            for name, tensor in state_dict.items():
                if name.endswith(".linear_attn.in_proj_ba.weight"):
                    base = name[: -len("in_proj_ba.weight")]
                    renamed[base + "in_proj_b.weight"] = tensor
                    renamed[base + "in_proj_a.weight"] = tensor
                elif name.endswith(".linear_attn.in_proj_qkvz.weight"):
                    # vLLM 按 in_proj_qkv(shard 0,1,2)+ in_proj_z(shard 3)分片加载;
                    # GGUF 的 attn_qkv→in_proj_qkv、attn_gate→in_proj_z
                    base = name[: -len("in_proj_qkvz.weight")]
                    renamed[base + "in_proj_qkv.weight"] = tensor
                    renamed[base + "in_proj_z.weight"] = tensor
                else:
                    renamed[name] = tensor
            state_dict = renamed
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            # Reverse HF convention: mlp -> block_sparse_moe
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            """
            Map HuggingFace parameter name to GGUF tensor name.

            This function handles the mismatch between HF parameter naming
            conventions and gguf-py's expected format:
            1. Strips 'model.' prefix (common in multimodal models)
            2. Converts '_weight' suffix to '.weight' (Gemma3 compatibility)
            3. Searches vision_name_map for multimodal parameters
            4. Falls back to text_name_map for language model parameters

            Args:
                hf_name: Full HuggingFace parameter name (e.g.,
                        'model.multi_modal_projector.mm_soft_emb_norm.weight')

            Returns:
                GGUF tensor name with suffix (e.g., 'mm.soft_emb_norm.weight')
                or None if no mapping found
            """
            # In transformers v5, multimodal models (e.g. Gemma3) wrap
            # all sub-models under an outer 'model.' attribute, producing
            # state_dict keys like 'model.language_model.layers.0...' and
            # 'model.vision_tower.vision_model...'.  Strip this outer
            # prefix so the keys match what gguf-py expects.
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]  # Remove outer 'model.'

            # Strip 'language_model.' prefix for multimodal models - gguf-py
            # tensor mappings expect parameter names without this prefix.
            # Note: 'model.' prefix should be KEPT for text-only models as
            # gguf-py expects it.
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]  # Remove 'language_model.'
                # Re-add 'model.' prefix because gguf-py text tensor maps
                # expect 'model.layers...' format.
                if is_multimodal:
                    hf_name = "model." + hf_name

            # Parse parameter name and suffix
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                # Handle '_weight' suffix (Gemma3 naming: parameter ends with
                # '_weight' instead of '.weight')
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]  # Remove '_weight'
                    suffix = "weight"

            gguf_name = None
            # Priority 1: Search vision/projector parameters for multimodal models
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)

            # Priority 2: Search text backbone parameters
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)

            if gguf_name is None:
                return None

            # suffix 为空(裸参数如 A_log/dt_bias)时不能加尾点,
            # 否则映射键 'blk.N.ssm_a.' 与 GGUF tensor 名 'blk.N.ssm_a' 不匹配;
            # 但以 bias 结尾的裸参数(如 dt_bias)对应的 GGUF tensor 名带 .bias
            if not suffix:
                if base_name.endswith("bias"):
                    return gguf_name + ".bias"
                return gguf_name
            return gguf_name + "." + suffix

        # Build mapping and track unmapped parameters
        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)

            # Track mapping success
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                # Parameter not in manual overrides either
                unmapped_params.append(hf_name)

        # All parameters (except those initialized by other means) must be mapped:
        # both vision/projector and backbone
        if unmapped_params:
            unmapped_params = list(
                filter(
                    lambda x: not any(re.fullmatch(p, x) for p in sideload_params),
                    unmapped_params,
                )
            )
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): "
                f"{unmapped_params}"
            )
        return gguf_to_hf_name_map

    def _get_gguf_weight_type(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        gguf_files = self._get_all_gguf_files(model_name_or_path)
        weight_type_map = {}
        for f in gguf_files:
            weight_type_map.update(get_gguf_weight_type_map(f, gguf_to_hf_name_map))
        is_multimodal = detect_gguf_multimodal(model_name_or_path) is not None
        if is_multimodal:
            mmproj_file = detect_gguf_multimodal(model_name_or_path)
            assert mmproj_file is not None, (
                "Could not find mm_proj file for multimodal GGUF model"
            )
            logger.info("Loading extra mm_proj weights from %s...", mmproj_file)
            mm_proj_weight_type_map = get_gguf_weight_type_map(
                mmproj_file, gguf_to_hf_name_map
            )
            weight_type_map.update(mm_proj_weight_type_map)
        return weight_type_map

    def _get_weights_iterator(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Iterate over GGUF model weights, loading from both main model file and
        mmproj.gguf for multimodal Gemma3 models.

        For Gemma3 multimodal GGUF models:
        - Main file (gemma-3-*.gguf): Language model weights (model.*)
        - mmproj file (mmproj*.gguf): Vision tower + projector weights (v.*, mm.*)

        Yields:
            Tuples of (parameter_name, tensor) for all model weights
        """
        hf_config = model_config.hf_config
        is_multimodal = detect_gguf_multimodal(model_name_or_path) is not None

        if is_multimodal:
            # Load mm_proj (mm_encoder + projector) for multimodal weights
            mmproj_file = detect_gguf_multimodal(model_name_or_path)
            assert mmproj_file is not None, (
                "Could not find mm_proj file for multimodal GGUF model"
            )
            yield from gguf_quant_weights_iterator(mmproj_file, gguf_to_hf_name_map)

        gguf_files = self._get_all_gguf_files(model_name_or_path)
        if len(gguf_files) > 1:
            yield from gguf_quant_weights_iterator_multi(
                gguf_files, gguf_to_hf_name_map
            )
        else:
            base_iter = gguf_quant_weights_iterator(
                model_name_or_path, gguf_to_hf_name_map
            )
            yield from self._iter_lm_head_unquantized(base_iter)

    def _iter_lm_head_unquantized(
        self, base_iter: Generator[tuple[str, torch.Tensor], None, None]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """GGUF 的 lm_head(output.weight)常以量化类型存储(Q6_K 等),但
        ParallelLMHead 不支持 qweight_type 参数:拦截 qweight_type/qweight,
        反量化后以 lm_head.weight 产出。"""
        qweight_type: int | None = None
        embed_qwt_type: gguf.GGMLQuantizationType | None = None
        for name, tensor in base_iter:
            if name == "lm_head.qweight_type":
                qweight_type = tensor.item()
                continue
            if name == "model.embed_tokens.qweight_type":
                embed_qwt_type = tensor.item()
                continue
            if name == "lm_head.qweight":
                assert qweight_type is not None
                # ops.ggml_dequantize 仅 CUDA;CPU 上用 gguf.dequantize 反量化
                # 内存纪律:dequant 是 float32(248320x5120 ≈ 5GB),转 fp16 后立即释放,
                # 严禁缓存复用(曾致双表 10GB 驻留 + 双 worker → user cgroup OOM)
                dequant = gguf.dequantize(
                    tensor.numpy(), gguf.GGMLQuantizationType(qweight_type)
                )
                tensor = torch.from_numpy(dequant).to(torch.float16)
                del dequant
                name = "lm_head.weight"
            elif name == "model.embed_tokens.qweight":
                # llama.cpp 的 embed = token_embd.weight(Q4_K),lm_head = output.weight(Q6_K),
                # 两表不相关(corr≈0)。此前误复用 output.weight 做 embed,导致整个链从输入就错。
                # 现在正确反量化 token_embd.weight 作为 embed 表。
                assert embed_qwt_type is not None
                dequant = gguf.dequantize(
                    tensor.numpy(), gguf.GGMLQuantizationType(embed_qwt_type)
                )
                tensor = torch.from_numpy(dequant).to(torch.float16)
                del dequant
                name = "model.embed_tokens.weight"
            elif name.endswith(".linear_attn.conv1d.weight"):
                # vLLM GDN conv1d 是 [conv_dim, 1, kernel](unsqueeze 过),
                # GGUF 产出 [channels, kernel] 2D,补上中间维
                tensor = tensor.unsqueeze(1)
            elif name.endswith(".linear_attn.A_log"):
                # GGUF 的 ssm_a 是 llama.cpp 预计算的 -exp(A_log)(小负值),
                # vLLM forward 里再 -exp(A_log) 会双重 exp;转换回原始 log 值
                tensor = torch.log(-tensor.float())
            yield name, tensor

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        gguf_files = self._get_all_gguf_files(local_model_path)
        all_extra_names = []
        for f in gguf_files:
            all_extra_names.extend(get_gguf_extra_tensor_names(f, gguf_weights_map))
        if "lm_head.weight" in all_extra_names:
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map = self._get_gguf_weight_type(
            model_config, local_model_path, gguf_weights_map
        )
        # filter out unquantized modules to skip
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]
        # lm_head/output 在 GGUF 里常以量化类型存储(Q6_K 等),但 ParallelLMHead
        # 不支持 qweight_type 元数据,强制按未量化(反量化)加载
        if "lm_head" not in unquant_names:
            unquant_names.append("lm_head")
        # embed_tokens 同理:Qwen3_5 的 GGUFEmbeddingMethod 对 qweight_type
        # 处理有缺陷(参数未创建),强制反量化走普通 weight 加载
        if "model.embed_tokens" not in unquant_names:
            unquant_names.append("model.embed_tokens")
        logger.debug("GGUF unquantized modules: %s", unquant_names)
        if TYPE_CHECKING:
            vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
