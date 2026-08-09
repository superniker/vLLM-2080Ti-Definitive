# vLLM-2080Ti-dev 修改记录

基准:原始 fork weicj/vLLM-2080Ti-Definitive @ 0a0caa1(即 vllm-orig-2080Ti,git 干净)
目标:为 GGUF/Q4 路线添加的功能修复,全部以"默认不影响其他格式"为原则:
  - GGUF 专属逻辑用 `load_format == "gguf"` 或环境变量 `VLLM_GDN_GGUF_LAYOUT=1` 分流
  - 未设置时(默认)完全走原版路径,兼容 AWQ/GPTQ/FP16/BF16 等 vLLM 原生格式
  - 每处改动带 `[FORK 兼容]` 注释

验证流程(每移植一组后执行):
1. AWQ(原始 checkpoint):serve_awq.sh → "23*47=" 应为 "1081"
2. GGUF(Q4):serve_gguf.sh → "1+1=" 应为 "2"

---

## 移植记录

### [1] A 类:GGUF 配置/加载修复(4 文件,2026-08-09)
- `transformers_utils/config.py`:qwen35→Qwen3_5Config 映射;GGUF 主文件纯文本时 vision=None(走 Qwen3_5ForCausalLM)
  - 仅影响 model_type=="qwen35" 且 GGUF 加载路径;safetensors 多模态不受影响
- `transformers_utils/configs/qwen3_5.py`:GGUF 加载时顶层 kwargs 转发给 text_config
  - 仅 GGUF 路径(字段缺失时);原版行为(子配置 dict)不变
- `v1/core/kv_cache_utils.py`:mamba cache page_size_padded 填充修复
  - 仅 `mamba_cache_mode != "align"` 且层支持 padding 时生效;align 模式走原版
- `v1/attention/backends/flex_attention.py`:key/value_cache view→reshape(4 行)
  - reshape 比 view 宽容(非连续张量也可),数值等价,所有格式通用

### [2] qwen3_next.py:Qwen3NextRMSNorm 按 load_format 分流(2026-08-09)
- 新增 `_get_qwen3_next_rms_norm_cls(load_format)`:GGUF → 普通 RMSNorm(llama.cpp 权重含 +1);
  其他(safetensors/AWQ 等)→ GemmaRMSNorm(原版 1+w 语义)
- 删除 GDN 输入的 .float() 强制转换(原版 4 处):各格式走原生 dtype
  (GGUF FP32 / AWQ FP16),避免精度与 dtype 错配

### [3] B 类:GDN 内核双路线布局(GGUF mod16 vs AWQ div3)(2026-08-09)
- `fla/ops/fused_sigmoid_gating.py`:
  - 内核新增 `GGUF_LAYOUT: tl.constexpr` 参数
  - i_h 映射:GGUF_LAYOUT=True → `(i_hv + V_START) % H`(mod16,llama.cpp 布局);
    False(默认)→ `i_hv // (HV // H)`(div3,transformers/AWQ 原版)
  - 包装函数新增 `gguf_layout=False` 参数并透传内核
- `fla/ops/fused_recurrent.py`:
  - 两处内核 i_h = `i_hv % H`(mod16)。仅 GGUF 的 forward_native(chunk)路径使用;
    AWQ 等走 forward_cuda(fi_chunk_gated_delta_rule,flashinfer 内置 div3),不受影响
- `mamba/gdn_linear_attn.py`:
  - 模块级 `_GDN_GGUF_LAYOUT`(环境变量 `VLLM_GDN_GGUF_LAYOUT=1`,默认 False)
  - import `get_tp_group`(GGUF q/k all-gather 用)
  - `_gdn_v_start`(v-head 全局起始,仅 GGUF 且 TP>1 时非 0,默认 0)
  - prefill 非 spec 分支:GGUF → fused_post_conv_prep + q/k all-gather;
    AWQ/其他 → 原版 rearrange_mixed_qkv(无 gather)
  - decode split_non_spec:q/k all-gather 仅 GGUF
  - 内核调用(spec/decode)传 v_start(默认 0)+ gguf_layout(默认 False)
  - **prefill 主路径(2.3)双分支**:GGUF → `fused_sigmoid_gating_delta_rule_update` 直接内核
    (a_prefill/b_prefill,use_qk_l2norm=True,v_start/gguf_layout=True),且输出逐 token state
    须按 `last_recurrent_state[prefill_query_start_loc[1:]-1]` 切分取每序列末尾;
    AWQ/其他 → 原版 `self.chunk_gated_delta_rule`(chunk 式,输出即每序列末尾 state)
  - **if split_non_spec 分支 q/k gather 仅 GGUF**(`tp_size>1 and _GDN_GGUF_LAYOUT`);
    AWQ div3 半切直用不 gather(当前版无条件 gather 会破坏 AWQ,dev 已修正)
  - `_output_projection` dtype 分流:z 对齐 core dtype;GGUF FP32 模式(VLLM_GGUF_FP32=1)
    proj_in 保持 FP32,否则(AWQ)转 FP16;非量化层转 weight.dtype
  - `core_attn_out` 分配两处 dtype=torch.float32(GGUF FP32 链必需;AWQ 经 proj_in 分流回 FP16)
  - split_non_spec=False 时补 `a_prefill=a, b_prefill=b` 兜底(当前版会 NameError)

### [4] 验证(dev 工作副本,2026-08-09)
- [x] AWQ(dev,原始 checkpoint):1+1= → '2' ✓;23*47= → 1081 ✓;7*8-6= → '56' ✓
- [x] GGUF(dev,VLLM_GDN_GGUF_LAYOUT=1):1+1= → '2\n1+1=2\n' ✓(与旧版一致);
      23*47= → '1081(10' ✓;7*8-6= → '56-6=50' ✓
- [x] **双兼容达成**:同一份代码,GGUF 走 mod16 直接内核 + gather,AWQ 走 div3 chunk 原版路径
- 注:当前版(Definitive)的"AWQ 正常"实为 orig 目录功劳(其无条件直接内核+gather 会破坏 AWQ);
  dev 是首个真正双兼容的版本
