#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from typing import Any

import requests


CJK_RE = re.compile(r"[\u3400-\u9fff]")
REPEATED_CHARS = re.compile(r"(.)\1{15,}", re.DOTALL)


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def request_chat(
    base_url: str,
    model: str,
    payload: dict[str, Any],
    *,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        timeout=(30, timeout),
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), elapsed


def extract_message(data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    choices = data.get("choices") or []
    if not choices:
        raise AssertionError("response has no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    content = flatten_content(message.get("content") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    if not content.strip():
        raise AssertionError("response content is empty")
    return content, finish_reason, message


def find_repeated_ngram(text: str, *, n: int, repeats: int) -> str | None:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n * repeats:
        return None
    for idx in range(0, len(compact) - (n * repeats) + 1):
        gram = compact[idx : idx + n]
        if len(set(gram)) <= 1:
            continue
        if compact[idx : idx + (n * repeats)] == gram * repeats:
            return gram
    return None


def assert_not_degenerate(label: str, text: str) -> None:
    if REPEATED_CHARS.search(text):
        raise AssertionError(f"{label}: repeated character pattern detected")
    for n, repeats in ((4, 4), (6, 3)):
        gram = find_repeated_ngram(text, n=n, repeats=repeats)
        if gram is not None:
            raise AssertionError(f"{label}: repeated n-gram detected: {gram!r}")


def count_cjk(text: str) -> int:
    return len(CJK_RE.findall(text))


def assert_chinese_text(label: str, text: str, *, min_cjk: int) -> None:
    cjk = count_cjk(text)
    if cjk < min_cjk:
        raise AssertionError(f"{label}: too few Chinese characters: {cjk}")
    ratio = cjk / max(len(text), 1)
    if ratio < 0.25:
        raise AssertionError(f"{label}: Chinese character ratio too low: {ratio:.3f}")


def validate_cn_explain(text: str) -> dict[str, Any]:
    assert_not_degenerate("cn_explain", text)
    assert_chinese_text("cn_explain", text, min_cjk=18)
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 24:
        raise AssertionError("cn_explain: output too short")
    return {"cjk_chars": count_cjk(text), "text_chars": len(text)}


def normalize_scalar(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(normalize_scalar(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key}:{normalize_scalar(val)}" for key, val in sorted(value.items()))
    return str(value)


def validate_cn_json(text: str) -> dict[str, Any]:
    assert_not_degenerate("cn_json_extract", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"cn_json_extract: invalid JSON: {exc}") from exc

    for key in ("person", "city", "gpu_count", "gpu_model"):
        if key not in payload:
            raise AssertionError(f"cn_json_extract: missing key {key!r}")

    person = normalize_scalar(payload["person"])
    city = normalize_scalar(payload["city"])
    gpu_count = normalize_scalar(payload["gpu_count"])
    gpu_model = normalize_scalar(payload["gpu_model"]).lower()

    if "张敏" not in person:
        raise AssertionError("cn_json_extract: person field does not contain 张敏")
    if "杭州" not in city:
        raise AssertionError("cn_json_extract: city field does not contain 杭州")
    if not any(token in gpu_count for token in ("2", "两")):
        raise AssertionError("cn_json_extract: gpu_count does not describe two GPUs")
    if "2080" not in gpu_model:
        raise AssertionError("cn_json_extract: gpu_model does not mention 2080")

    return {"parsed_json": payload}


def validate_cn_keywords(text: str) -> dict[str, Any]:
    assert_not_degenerate("cn_keywords", text)
    assert_chinese_text("cn_keywords", text, min_cjk=24)
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 40:
        raise AssertionError("cn_keywords: output too short")
    clauses = [part.strip() for part in re.split(r"[。！？；\n]+", text) if part.strip()]
    if len(clauses) < 2:
        raise AssertionError("cn_keywords: expected multiple clauses or sentences")
    for keyword in ("质量", "吞吐", "重复"):
        if keyword not in text:
            raise AssertionError(f"cn_keywords: missing keyword {keyword}")
    return {"cjk_chars": count_cjk(text), "clauses": len(clauses), "text_chars": len(text)}


def build_cases(model: str, max_tokens: int, temperature: float, seed: int) -> list[dict[str, Any]]:
    base = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return [
        {
            "label": "cn_explain",
            "payload": {
                **base,
                "messages": [
                    {
                        "role": "user",
                        "content": "请用中文两句话解释，为什么服务模式回归时要先做质量 smoke，再看 decode 吞吐。不要分点，不要重复。",
                    }
                ],
            },
            "validator": validate_cn_explain,
        },
        {
            "label": "cn_json_extract",
            "payload": {
                **base,
                "messages": [
                    {
                        "role": "user",
                        "content": "阅读这句话：张敏在杭州用两张 RTX 2080 Ti 测试 Qwen3.6 27B。只返回一个 JSON 对象，包含 person, city, gpu_count, gpu_model 四个字段。",
                    }
                ],
                "response_format": {"type": "json_object"},
            },
            "validator": validate_cn_json,
        },
        {
            "label": "cn_keywords",
            "payload": {
                **base,
                "messages": [
                    {
                        "role": "user",
                        "content": "用中文写一段 80 到 120 字的说明，主题是同一路径既要测质量也要测吞吐。必须包含“质量”“吞吐”“重复”三个词，不要列表，不要重复同一句话。",
                    }
                ],
            },
            "validator": validate_cn_keywords,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    results = []
    ok = True
    for case in build_cases(args.model, args.max_tokens, args.temperature, args.seed):
        result: dict[str, Any] = {"label": case["label"], "ok": False}
        data: dict[str, Any] | None = None
        content = ""
        finish_reason = ""
        message: dict[str, Any] = {}
        try:
            data, elapsed = request_chat(
                args.base_url,
                args.model,
                case["payload"],
                timeout=args.timeout,
            )
            content, finish_reason, message = extract_message(data)
            details = case["validator"](content)
            result.update(
                {
                    "ok": True,
                    "elapsed_s": elapsed,
                    "finish_reason": finish_reason,
                    "content_preview": content[:240],
                    "message_preview": json.dumps(message, ensure_ascii=False)[:800],
                    "usage": data.get("usage"),
                    **details,
                }
            )
        except Exception as exc:  # noqa: BLE001
            ok = False
            result["error"] = f"{type(exc).__name__}: {exc}"
            if finish_reason:
                result["finish_reason"] = finish_reason
            if content:
                result["content_preview"] = content[:240]
            if message:
                result["message_preview"] = json.dumps(message, ensure_ascii=False)[:800]
            if data is not None:
                result["raw_response_preview"] = json.dumps(data, ensure_ascii=False)[:1200]
                if data.get("usage") is not None:
                    result["usage"] = data.get("usage")
        results.append(result)

    output = {"ok": ok, "results": results}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
