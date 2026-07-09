#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
REMOTE_HOST=${REMOTE_HOST:?set REMOTE_HOST}
REMOTE_ROOT=${REMOTE_ROOT:?set REMOTE_ROOT}
REMOTE_USER=${REMOTE_USER:?set REMOTE_USER}
GPU_DEVICES=${GPU_DEVICES:-1,2}
PORT=${PORT:-19447}
START_TIMEOUT=${START_TIMEOUT:-300}
WARMUPS=${WARMUPS:-1}
MEASURED_RUNS=${MEASURED_RUNS:-3}
READ_TIMEOUT=${READ_TIMEOUT:-1800}
QUALITY_MAX_TOKENS=${QUALITY_MAX_TOKENS:-160}
QUALITY_TIMEOUT=${QUALITY_TIMEOUT:-240}
LANES=${LANES:-4096:128,65536:512}
CASE_FILTER=${CASE_FILTER:-}
SPEC_SYNC_OVERRIDE=${SPEC_SYNC_OVERRIDE:-}
MTP_OVERRIDE=${MTP_OVERRIDE:-}
COMPILATION_CONFIG_OVERRIDE=${COMPILATION_CONFIG_OVERRIDE:-}
ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE=${ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE:-}
INT8KV_DIRECT_PAGED_OVERRIDE=${INT8KV_DIRECT_PAGED_OVERRIDE:-}
INT8KV_ALLOW_3D_DECODE_OVERRIDE=${INT8KV_ALLOW_3D_DECODE_OVERRIDE:-}
INT8KV_CONTINUATION_MIN_Q_OVERRIDE=${INT8KV_CONTINUATION_MIN_Q_OVERRIDE:-}
INT8KV_CONTINUATION_DEQUANT_OVERRIDE=${INT8KV_CONTINUATION_DEQUANT_OVERRIDE:-}
INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE=${INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE:-}
INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE=${INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE:-}
WITH_COMPARE=${WITH_COMPARE:-1}
STRICT_PROMOTION=${STRICT_PROMOTION:-0}
PROMOTION_SHORT_LANE=${PROMOTION_SHORT_LANE:-4096:128}
FAST_GUARD_FP8=${FAST_GUARD_FP8:-70}
FAST_GUARD_INT4=${FAST_GUARD_INT4:-90}
RESULT_STAMP=${RESULT_STAMP:-$(date +%Y%m%d-%H%M%S)}
REMOTE_RESULT_DIR=${REMOTE_RESULT_DIR:-$REMOTE_ROOT/results/int8kv_eval_$RESULT_STAMP}
MAX_MODEL_LEN_OVERRIDE=${MAX_MODEL_LEN_OVERRIDE:-}
FP8_MAX_MODEL_LEN_OVERRIDE=${FP8_MAX_MODEL_LEN_OVERRIDE:-}
INT4_MAX_MODEL_LEN_OVERRIDE=${INT4_MAX_MODEL_LEN_OVERRIDE:-}
GPU_UTIL_OVERRIDE=${GPU_UTIL_OVERRIDE:-}
MAX_BATCHED_TOKENS_OVERRIDE=${MAX_BATCHED_TOKENS_OVERRIDE:-}
FP8_MODEL_DIR=${FP8_MODEL_DIR:?set FP8_MODEL_DIR}
INT4_MODEL_DIR=${INT4_MODEL_DIR:?set INT4_MODEL_DIR}

if [[ "$(id -u)" == "0" ]]; then
  LOCAL_EXEC_USER=${LOCAL_EXEC_USER:-${SUDO_USER:-}}
  if [[ -z "$LOCAL_EXEC_USER" ]]; then
    LOCAL_EXEC_USER=$(stat -c %U "$ROOT" 2>/dev/null || true)
  fi
  if [[ -n "$LOCAL_EXEC_USER" && "$LOCAL_EXEC_USER" != "root" ]]; then
    SSH=(runuser -u "$LOCAL_EXEC_USER" -- ssh)
    RSYNC=(runuser -u "$LOCAL_EXEC_USER" -- rsync)
  else
    SSH=(ssh)
    RSYNC=(rsync)
  fi
else
  SSH=(ssh)
  RSYNC=(rsync)
fi

run_ssh() {
  "${SSH[@]}" -o ConnectTimeout=10 "$REMOTE_HOST" "$@"
}

sync_runtime() {
  if [[ "${EVAL_SYNC:-1}" != "1" ]]; then
    return 0
  fi

  "${RSYNC[@]}" -a --delete "$ROOT/profiles/" "$REMOTE_HOST:$REMOTE_ROOT/profiles/"
  "${RSYNC[@]}" -a --delete "$ROOT/tools/" "$REMOTE_HOST:$REMOTE_ROOT/tools/"
  (
    cd "$ROOT"
    "${RSYNC[@]}" -aR \
      launcher.sh \
      build.sh \
      vllm/config/compilation.py \
      vllm/envs.py \
      vllm/v1/attention/backends/triton_attn.py \
      vllm/v1/attention/ops/triton_unified_attention.py \
      vllm/v1/worker/gpu_worker.py \
      vllm/v1/worker/gpu_model_runner.py \
      "$REMOTE_HOST:$REMOTE_ROOT/"
  )

  run_ssh "chown -R $REMOTE_USER:$REMOTE_USER '$REMOTE_ROOT/profiles' '$REMOTE_ROOT/tools' '$REMOTE_ROOT/launcher.sh' '$REMOTE_ROOT/build.sh' '$REMOTE_ROOT/vllm/config/compilation.py' '$REMOTE_ROOT/vllm/envs.py' '$REMOTE_ROOT/vllm/v1/attention/backends/triton_attn.py' '$REMOTE_ROOT/vllm/v1/attention/ops/triton_unified_attention.py' '$REMOTE_ROOT/vllm/v1/worker/gpu_worker.py' '$REMOTE_ROOT/vllm/v1/worker/gpu_model_runner.py'"

  run_ssh "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_USER='$REMOTE_USER' bash -s" <<'REMOTE_SYNC'
set -euo pipefail
site_packages=$(runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)
site_vllm="$site_packages/vllm"
if [[ -d "$site_vllm" ]]; then
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/config/compilation.py" \
    "$site_vllm/config/compilation.py"
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/envs.py" \
    "$site_vllm/envs.py"
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/v1/attention/backends/triton_attn.py" \
    "$site_vllm/v1/attention/backends/triton_attn.py"
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/v1/attention/ops/triton_unified_attention.py" \
    "$site_vllm/v1/attention/ops/triton_unified_attention.py"
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/v1/worker/gpu_worker.py" \
    "$site_vllm/v1/worker/gpu_worker.py"
  install -o "$REMOTE_USER" -g "$REMOTE_USER" -m 0644 \
    "$REMOTE_ROOT/vllm/v1/worker/gpu_model_runner.py" \
    "$site_vllm/v1/worker/gpu_model_runner.py"
fi
REMOTE_SYNC
}

sync_runtime

run_ssh \
  "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_USER='$REMOTE_USER' GPU_DEVICES='$GPU_DEVICES' PORT='$PORT' START_TIMEOUT='$START_TIMEOUT' WARMUPS='$WARMUPS' MEASURED_RUNS='$MEASURED_RUNS' READ_TIMEOUT='$READ_TIMEOUT' QUALITY_MAX_TOKENS='$QUALITY_MAX_TOKENS' QUALITY_TIMEOUT='$QUALITY_TIMEOUT' LANES='$LANES' CASE_FILTER='$CASE_FILTER' SPEC_SYNC_OVERRIDE='$SPEC_SYNC_OVERRIDE' MTP_OVERRIDE='$MTP_OVERRIDE' COMPILATION_CONFIG_OVERRIDE='$COMPILATION_CONFIG_OVERRIDE' ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE='$ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE' INT8KV_DIRECT_PAGED_OVERRIDE='$INT8KV_DIRECT_PAGED_OVERRIDE' INT8KV_ALLOW_3D_DECODE_OVERRIDE='$INT8KV_ALLOW_3D_DECODE_OVERRIDE' INT8KV_CONTINUATION_MIN_Q_OVERRIDE='$INT8KV_CONTINUATION_MIN_Q_OVERRIDE' INT8KV_CONTINUATION_DEQUANT_OVERRIDE='$INT8KV_CONTINUATION_DEQUANT_OVERRIDE' INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE='$INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE' INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE='$INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE' WITH_COMPARE='$WITH_COMPARE' STRICT_PROMOTION='$STRICT_PROMOTION' PROMOTION_SHORT_LANE='$PROMOTION_SHORT_LANE' FAST_GUARD_FP8='$FAST_GUARD_FP8' FAST_GUARD_INT4='$FAST_GUARD_INT4' REMOTE_RESULT_DIR='$REMOTE_RESULT_DIR' MAX_MODEL_LEN_OVERRIDE='$MAX_MODEL_LEN_OVERRIDE' FP8_MAX_MODEL_LEN_OVERRIDE='$FP8_MAX_MODEL_LEN_OVERRIDE' INT4_MAX_MODEL_LEN_OVERRIDE='$INT4_MAX_MODEL_LEN_OVERRIDE' GPU_UTIL_OVERRIDE='$GPU_UTIL_OVERRIDE' MAX_BATCHED_TOKENS_OVERRIDE='$MAX_BATCHED_TOKENS_OVERRIDE' FP8_MODEL_DIR='$FP8_MODEL_DIR' INT4_MODEL_DIR='$INT4_MODEL_DIR' bash -s" <<'REMOTE'
set -euo pipefail

cd "$REMOTE_ROOT"
mkdir -p "$REMOTE_RESULT_DIR"
chown -R "$REMOTE_USER:$REMOTE_USER" "$REMOTE_RESULT_DIR"

read_profile_value() {
  local file=$1
  local key=$2
  awk -F= -v key="$key" '
    $1 == key {
      value = substr($0, index($0, "=") + 1)
      gsub(/^[ \t]+|[ \t]+$/, "", value)
      gsub(/^'\''|'\''$/, "", value)
      gsub(/^"|"$/, "", value)
      print value
      exit
    }
  ' "$file"
}

set_profile_value() {
  local file=$1
  local key=$2
  local value=$3
  local escaped=$value
  escaped=${escaped//\\/\\\\}
  escaped=${escaped//&/\\&}
  escaped=${escaped//|/\\|}
  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${escaped}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

sanitize() {
  printf '%s' "$1" | tr '/ ' '__' | tr -c 'A-Za-z0-9_.-' '_'
}

reset_prefix_cache() {
  runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - "$PORT" <<'PY'
import sys

import requests

port = sys.argv[1]
response = requests.post(
    f"http://127.0.0.1:{port}/reset_prefix_cache",
    timeout=(10, 60),
)
response.raise_for_status()
PY
}

stop_runtime_vllm() {
  local pid_file pid cmd
  while IFS= read -r pid_file; do
    [[ -n "$pid_file" ]] || continue
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [[ -n "$pid" && -d "/proc/$pid" ]]; then
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
      if [[ "$cmd" == *"$REMOTE_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server"* ]]; then
        kill "$pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pid_file"
  done < <(find "$REMOTE_ROOT/run-logs" -maxdepth 1 -type f -name '*.pid' -print 2>/dev/null | sort)

  sleep 2
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f "$REMOTE_ROOT/.venv/bin/python -m vllm.entrypoints.openai.api_server" || true)

  sleep 2
  cleanup_orphan_vllm_gpu_workers
}

cleanup_orphan_vllm_gpu_workers() {
  local selected_csv line idx uuid gpu_uuid pid pname ppid comm
  declare -A selected_uuids=()

  selected_csv=",$GPU_DEVICES,"
  while IFS=',' read -r idx uuid; do
    idx=${idx// /}
    uuid=${uuid// /}
    [[ -n "$idx" && -n "$uuid" ]] || continue
    if [[ "$selected_csv" == *",$idx,"* ]]; then
      selected_uuids["$uuid"]=1
    fi
  done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null || true)

  while IFS=',' read -r gpu_uuid pid pname; do
    gpu_uuid=${gpu_uuid// /}
    pid=${pid// /}
    pname=${pname## }
    [[ -n "$gpu_uuid" && -n "$pid" ]] || continue
    [[ -n "${selected_uuids[$gpu_uuid]:-}" ]] || continue
    [[ -d "/proc/$pid" ]] || continue

    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    if [[ "$ppid" != "1" ]]; then
      continue
    fi
    case "$pname $comm" in
      *VLLM::Worker_*|*VLLM::EngineCore*|*VLLM::EngineCoreClient*)
        kill -TERM "$pid" 2>/dev/null || true
        ;;
    esac
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)

  sleep 1
  while IFS=',' read -r gpu_uuid pid pname; do
    gpu_uuid=${gpu_uuid// /}
    pid=${pid// /}
    pname=${pname## }
    [[ -n "$gpu_uuid" && -n "$pid" ]] || continue
    [[ -n "${selected_uuids[$gpu_uuid]:-}" ]] || continue
    [[ -d "/proc/$pid" ]] || continue

    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    comm=$(cat "/proc/$pid/comm" 2>/dev/null || true)
    if [[ "$ppid" != "1" ]]; then
      continue
    fi
    case "$pname $comm" in
      *VLLM::Worker_*|*VLLM::EngineCore*|*VLLM::EngineCoreClient*)
        kill -KILL "$pid" 2>/dev/null || true
        ;;
    esac
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits 2>/dev/null || true)
}

trap stop_runtime_vllm EXIT

make_eval_profile() {
  local rel=$1
  local mode=$2
  local out=$3
  local served_name model_variant max_len_override
  cp "$REMOTE_ROOT/profiles/$rel" "$out"
  set_profile_value "$out" COMPATIBLE_MODES "safe,normal,fast"
  if [[ -n "$MTP_OVERRIDE" ]]; then
    set_profile_value "$out" MTP_K "$MTP_OVERRIDE"
  fi
  model_variant=$(read_profile_value "$out" MODEL_VARIANT)
  max_len_override=${MAX_MODEL_LEN_OVERRIDE:-}
  if [[ -z "$max_len_override" ]]; then
    case "$model_variant" in
      fp8)
        max_len_override=${FP8_MAX_MODEL_LEN_OVERRIDE:-}
        ;;
      int4)
        max_len_override=${INT4_MAX_MODEL_LEN_OVERRIDE:-}
        ;;
    esac
  fi
  if [[ -n "$max_len_override" ]]; then
    set_profile_value "$out" MAX_MODEL_LEN "$max_len_override"
  fi
  if [[ -n "$GPU_UTIL_OVERRIDE" ]]; then
    set_profile_value "$out" GPU_UTIL "$GPU_UTIL_OVERRIDE"
  fi
  if [[ -n "$MAX_BATCHED_TOKENS_OVERRIDE" ]]; then
    set_profile_value "$out" MAX_BATCHED_TOKENS "$MAX_BATCHED_TOKENS_OVERRIDE"
  fi
  served_name=$(read_profile_value "$out" SERVED_NAME)
  if [[ -n "$served_name" ]]; then
    set_profile_value "$out" SERVED_NAME "${served_name}-${mode}-eval"
  fi
}

run_one_case() {
  local group=$1
  local profile=$2
  local model_dir=$3
  local mode=$4
  local role=$5
  local fast_guard=$6

  local case_id case_dir tmp_profile served_name launch_out launch_err
  local launch_env=()
  case_id=$(sanitize "${group}_${mode}")
  case_dir="$REMOTE_RESULT_DIR/$case_id"
  mkdir -p "$case_dir"
  chown -R "$REMOTE_USER:$REMOTE_USER" "$case_dir"
  tmp_profile="$case_dir/profile.env"
  make_eval_profile "$profile" "$mode" "$tmp_profile"
  if [[ -n "$SPEC_SYNC_OVERRIDE" ]]; then
    launch_env+=("VLLM_SM75_SPEC_SYNC_MODE_OVERRIDE=$SPEC_SYNC_OVERRIDE")
  fi
  if [[ -n "$ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE" ]]; then
    launch_env+=("VLLM_ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH=$ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE")
  fi
  if [[ -n "$COMPILATION_CONFIG_OVERRIDE" ]]; then
    launch_env+=("COMPILATION_CONFIG_JSON=$COMPILATION_CONFIG_OVERRIDE")
  fi
  if [[ -n "$INT8KV_DIRECT_PAGED_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_FA_DIRECT_PAGED=$INT8KV_DIRECT_PAGED_OVERRIDE")
  fi
  if [[ -n "$INT8KV_ALLOW_3D_DECODE_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_ALLOW_3D_DECODE=$INT8KV_ALLOW_3D_DECODE_OVERRIDE")
  fi
  if [[ -n "$INT8KV_CONTINUATION_MIN_Q_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_FA_CONTINUATION_MIN_Q=$INT8KV_CONTINUATION_MIN_Q_OVERRIDE")
  fi
  if [[ -n "$INT8KV_CONTINUATION_DEQUANT_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_FA_CONTINUATION_DEQUANT=$INT8KV_CONTINUATION_DEQUANT_OVERRIDE")
  fi
  if [[ -n "$INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_FA_CONTINUATION_INCREMENTAL_REUSE=$INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE")
  fi
  if [[ -n "$INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE" ]]; then
    launch_env+=("VLLM_INT8KV_FA_CONTINUATION_FP16_DEQUANT=$INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE")
  fi
  chown "$REMOTE_USER:$REMOTE_USER" "$tmp_profile"
  served_name=$(read_profile_value "$tmp_profile" SERVED_NAME)
  launch_out="$case_dir/launch.out"
  launch_err="$case_dir/launch.err"
  : >"$launch_out"
  : >"$launch_err"
  chown "$REMOTE_USER:$REMOTE_USER" "$launch_out" "$launch_err"

  {
    printf 'group\t%s\nprofile\t%s\nmode\t%s\nrole\t%s\nfast_guard\t%s\nmodel_dir\t%s\nserved_name\t%s\n' \
      "$group" "$profile" "$mode" "$role" "$fast_guard" "$model_dir" "$served_name"
    printf 'spec_sync_override\t%s\nallow_mamba_spec_full_cudagraph_override\t%s\ncompilation_config_override\t%s\n' \
      "$SPEC_SYNC_OVERRIDE" "$ALLOW_MAMBA_SPEC_FULL_CUDAGRAPH_OVERRIDE" "$COMPILATION_CONFIG_OVERRIDE"
    printf 'int8kv_direct_paged_override\t%s\nint8kv_allow_3d_decode_override\t%s\nint8kv_continuation_min_q_override\t%s\nint8kv_continuation_dequant_override\t%s\nint8kv_continuation_incremental_reuse_override\t%s\nint8kv_continuation_fp16_dequant_override\t%s\n' \
      "$INT8KV_DIRECT_PAGED_OVERRIDE" "$INT8KV_ALLOW_3D_DECODE_OVERRIDE" "$INT8KV_CONTINUATION_MIN_Q_OVERRIDE" "$INT8KV_CONTINUATION_DEQUANT_OVERRIDE" "$INT8KV_CONTINUATION_INCREMENTAL_REUSE_OVERRIDE" "$INT8KV_CONTINUATION_FP16_DEQUANT_OVERRIDE"
  } >"$case_dir/meta.tsv"
  chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/meta.tsv"

  stop_runtime_vllm

  runuser -u "$REMOTE_USER" -- env \
    MODEL_DIR="$model_dir" \
    PROFILE_FILE="$tmp_profile" \
    MODE="$mode" \
    GPU_DEVICES="$GPU_DEVICES" \
    PORT="$PORT" \
    SERVICE_SCOPE=local \
    NON_INTERACTIVE=1 \
    VLLM_SERVER_DEV_MODE=1 \
    "${launch_env[@]}" \
    bash -lc "cd '$REMOTE_ROOT' && ./launcher.sh --print-config" \
    >"$case_dir/print-config.out" 2>"$case_dir/print-config.err" || true
  chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/print-config.out" "$case_dir/print-config.err"

  if ! runuser -u "$REMOTE_USER" -- env \
    MODEL_DIR="$model_dir" \
    PROFILE_FILE="$tmp_profile" \
    MODE="$mode" \
    GPU_DEVICES="$GPU_DEVICES" \
    PORT="$PORT" \
    SERVICE_SCOPE=local \
    NON_INTERACTIVE=1 \
    START_TIMEOUT="$START_TIMEOUT" \
    VLLM_SERVER_DEV_MODE=1 \
    "${launch_env[@]}" \
    bash -lc "cd '$REMOTE_ROOT' && ./launcher.sh --non-interactive" \
    >>"$launch_out" 2>>"$launch_err"; then
    printf 'launch_failed\n' >"$case_dir/status"
    chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/status"
    return 0
  fi

  if runuser -u "$REMOTE_USER" -- bash -lc \
    "cd '$REMOTE_ROOT' && .venv/bin/python tools/int8kv_quality_smoke.py --base-url 'http://127.0.0.1:$PORT/v1' --model '$served_name' --max-tokens '$QUALITY_MAX_TOKENS' --timeout '$QUALITY_TIMEOUT' --json-out '$case_dir/quality.json'" \
    >"$case_dir/quality.out" 2>"$case_dir/quality.err"; then
    printf 'quality_ok\n' >"$case_dir/quality.status"
  else
    printf 'quality_failed\n' >"$case_dir/quality.status"
  fi
  chown "$REMOTE_USER:$REMOTE_USER" \
    "$case_dir/quality.json" "$case_dir/quality.out" "$case_dir/quality.err" "$case_dir/quality.status" 2>/dev/null || true
  reset_prefix_cache >>"$case_dir/quality.out" 2>>"$case_dir/quality.err" || true

  IFS=',' read -r -a lane_specs <<<"$LANES"
  for lane in "${lane_specs[@]}"; do
    [[ -n "$lane" ]] || continue
    local prompt_tokens gen_tokens out_json bench_out bench_err run_index label
    prompt_tokens=${lane%%:*}
    gen_tokens=${lane##*:}
    out_json="$case_dir/pp${prompt_tokens}_tg${gen_tokens}.jsonl"
    bench_out="$case_dir/pp${prompt_tokens}_tg${gen_tokens}.out"
    bench_err="$case_dir/pp${prompt_tokens}_tg${gen_tokens}.err"
    : >"$bench_out"
    : >"$bench_err"
    chown "$REMOTE_USER:$REMOTE_USER" "$bench_out" "$bench_err"
    for ((run_index = 0; run_index < WARMUPS + MEASURED_RUNS; run_index += 1)); do
      if (( run_index < WARMUPS )); then
        label="${case_id}-pp${prompt_tokens}-tg${gen_tokens}-warmup$((run_index + 1))"
      else
        label="${case_id}-pp${prompt_tokens}-tg${gen_tokens}-run$((run_index - WARMUPS + 1))"
      fi
      runuser -u "$REMOTE_USER" -- bash -lc \
        "cd '$REMOTE_ROOT' && .venv/bin/python tools/profile_request.py --model-dir '$model_dir' --served-name '$served_name' --base-url 'http://127.0.0.1:$PORT/v1' --endpoint completions --prompt-tokens '$prompt_tokens' --gen-tokens '$gen_tokens' --label '$label' --out '$out_json' --ignore-eos --pure-filler --prompt-salt '$label' --read-timeout '$READ_TIMEOUT'" \
        >>"$bench_out" 2>>"$bench_err" || true
      reset_prefix_cache >>"$bench_out" 2>>"$bench_err" || true
    done
    chown "$REMOTE_USER:$REMOTE_USER" "$out_json" 2>/dev/null || true
  done

  printf 'measured\n' >"$case_dir/status"
  chown "$REMOTE_USER:$REMOTE_USER" "$case_dir/status"
  stop_runtime_vllm
}

cases_file="$REMOTE_RESULT_DIR/cases.tsv"
cat >"$cases_file" <<CASES
group	profile	model_dir	mode	role	fast_guard
fp8_int8kv	qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env	$FP8_MODEL_DIR	normal	int8_candidate	$FAST_GUARD_FP8
fp8_int8kv	qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env	$FP8_MODEL_DIR	fast	int8_candidate	$FAST_GUARD_FP8
int4_int8kv	qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env	$INT4_MODEL_DIR	normal	int8_candidate	$FAST_GUARD_INT4
int4_int8kv	qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env	$INT4_MODEL_DIR	fast	int8_candidate	$FAST_GUARD_INT4
CASES

if [[ "$WITH_COMPARE" == "1" ]]; then
  cat >>"$cases_file" <<CASES
fp8_fp16kv	qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env	$FP8_MODEL_DIR	normal	compare	0
fp8_tqk8v4	qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env	$FP8_MODEL_DIR	fast	compare	0
int4_fp16kv	qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env	$INT4_MODEL_DIR	normal	compare	0
int4_tqk8v4	qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env	$INT4_MODEL_DIR	fast	compare	0
CASES
fi

if [[ -n "$CASE_FILTER" ]]; then
  awk -F'\t' -v pat="$CASE_FILTER" '
    NR == 1 {
      print
      next
    }
    ($1 "\t" $2 "\t" $4 "\t" $5) ~ pat
  ' "$cases_file" >"$cases_file.filtered"
  mv "$cases_file.filtered" "$cases_file"
fi

tail -n +2 "$cases_file" | while IFS=$'\t' read -r group profile model_dir mode role fast_guard; do
  run_one_case "$group" "$profile" "$model_dir" "$mode" "$role" "$fast_guard"
done

runuser -u "$REMOTE_USER" -- "$REMOTE_ROOT/.venv/bin/python" - "$REMOTE_RESULT_DIR" "$LANES" "$MEASURED_RUNS" "$STRICT_PROMOTION" "$PROMOTION_SHORT_LANE" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
lanes = [item for item in sys.argv[2].split(",") if item]
expected_runs = int(sys.argv[3])
strict_promotion = sys.argv[4] == "1"
promotion_short_lane = sys.argv[5]


def sanitize(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.replace("/", "_"))
    return sanitized or "case"


def valid_filler(sample: str) -> bool:
    lower = sample.lower()
    return " the the the the" in lower and "climate change" not in lower and "introduction" not in lower


cases = []
with (root / "cases.tsv").open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        cases.append(row)

rows = []
failures: list[str] = []
by_group: dict[str, dict[str, dict[str, object]]] = {}

for case in cases:
    case_id = sanitize(f"{case['group']}_{case['mode']}")
    case_dir = root / case_id
    status = (case_dir / "status").read_text(encoding="utf-8", errors="ignore").strip() if (case_dir / "status").exists() else "missing"
    quality_path = case_dir / "quality.json"
    quality_ok = False
    quality_preview = ""
    if quality_path.exists():
      quality_data = json.loads(quality_path.read_text(encoding="utf-8"))
      quality_ok = bool(quality_data.get("ok"))
      previews = [item.get("content_preview", "") for item in quality_data.get("results", []) if isinstance(item, dict)]
      quality_preview = " | ".join(previews[:2])[:240]
    else:
      failures.append(f"{case['group']} {case['mode']}: missing quality.json")

    lane_stats: dict[str, dict[str, object]] = {}
    if status != "measured":
      failures.append(f"{case['group']} {case['mode']}: status={status}")
    if not quality_ok:
      failures.append(f"{case['group']} {case['mode']}: quality smoke failed")

    for lane in lanes:
      prompt_tokens, gen_tokens = lane.split(":")
      jsonl = case_dir / f"pp{prompt_tokens}_tg{gen_tokens}.jsonl"
      records = []
      if jsonl.exists():
          for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
              if not line.strip():
                  continue
              item = json.loads(line)
              if "-run" in str(item.get("label", "")):
                  records.append(item)

      prefill = [float(item["prefill_tok_s"]) for item in records if item.get("prefill_tok_s") is not None]
      decode = [float(item["decode_tok_s"]) for item in records if item.get("decode_tok_s") is not None]
      samples = [str(item.get("content_sample", "")) for item in records]
      filler_valid = bool(samples) and all(valid_filler(sample) for sample in samples)
      if len(records) < expected_runs:
          failures.append(
              f"{case['group']} {case['mode']} {lane}: runs={len(records)} expected={expected_runs}"
          )
      if not filler_valid:
          failures.append(f"{case['group']} {case['mode']} {lane}: filler drift")

      lane_stats[lane] = {
          "runs": len(records),
          "prefill_median": statistics.median(prefill) if prefill else None,
          "decode_median": statistics.median(decode) if decode else None,
          "filler_valid": filler_valid,
          "jsonl": str(jsonl),
      }

      rows.append(
          {
              "case_id": case_id,
              "group": case["group"],
              "profile": case["profile"],
              "mode": case["mode"],
              "role": case["role"],
              "fast_guard": case["fast_guard"],
              "status": status,
              "quality_ok": str(quality_ok),
              "quality_preview": quality_preview,
              "lane": lane,
              "runs": str(len(records)),
              "prefill_median": "" if not prefill else f"{statistics.median(prefill):.2f}",
              "decode_median": "" if not decode else f"{statistics.median(decode):.2f}",
              "filler_valid": str(filler_valid),
              "promotion_ready": "",
              "result": str(jsonl),
          }
      )

    by_group.setdefault(case["group"], {})[case["mode"]] = {
        "role": case["role"],
        "quality_ok": quality_ok,
        "fast_guard": float(case["fast_guard"] or 0),
        "lanes": lane_stats,
    }

promotion_rows: dict[tuple[str, str], str] = {}
for group, modes in by_group.items():
    if not all(name in modes for name in ("normal", "fast")):
        continue
    if modes["normal"]["role"] != "int8_candidate" or modes["fast"]["role"] != "int8_candidate":
        continue
    normal_lane = modes["normal"]["lanes"].get(promotion_short_lane, {})
    fast_lane = modes["fast"]["lanes"].get(promotion_short_lane, {})
    normal_decode = normal_lane.get("decode_median")
    fast_decode = fast_lane.get("decode_median")
    fast_quality = bool(modes["fast"]["quality_ok"])
    fast_guard = float(modes["fast"]["fast_guard"])
    ready = (
        fast_quality
        and isinstance(normal_decode, (int, float))
        and isinstance(fast_decode, (int, float))
        and float(fast_decode) > float(normal_decode)
        and float(fast_decode) >= fast_guard
    )
    promotion_rows[(sanitize(f"{group}_fast"), promotion_short_lane)] = "True" if ready else "False"
    if strict_promotion and not ready:
        failures.append(
            f"{group}: promotion guard failed on {promotion_short_lane} "
            f"(normal={normal_decode}, fast={fast_decode}, threshold={fast_guard})"
        )

for row in rows:
    key = (row["case_id"], row["lane"])
    if key in promotion_rows:
        row["promotion_ready"] = promotion_rows[key]

summary = root / "summary.tsv"
fieldnames = [
    "case_id",
    "group",
    "profile",
    "mode",
    "role",
    "fast_guard",
    "status",
    "quality_ok",
    "quality_preview",
    "lane",
    "runs",
    "prefill_median",
    "decode_median",
    "filler_valid",
    "promotion_ready",
    "result",
]
with summary.open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

verdict = root / "verdict.txt"
if failures:
    verdict.write_text("FAIL\n" + "\n".join(failures) + "\n", encoding="utf-8")
    print(f"FAIL {root}")
    print(verdict.read_text(encoding="utf-8"))
    raise SystemExit(1)

verdict.write_text("PASS\n", encoding="utf-8")
print(f"PASS {root}")
PY
REMOTE

echo "Remote result: $REMOTE_RESULT_DIR"
