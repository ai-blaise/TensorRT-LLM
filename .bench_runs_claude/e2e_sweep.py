# SPDX-License-Identifier: Apache-2.0
"""E2E serving sweep against the dynamo frontend: TTFT, steady tok/s/user,
aggregate tok/s across concurrencies. Streams completions, counts tokens via
usage (NextN delivers multi-token chunks, so chunk counting is wrong).

Usage: e2e_sweep.py [--url U] [--concurrencies 1,4,8,16,32] [--osl 256]
                    [--isl-text-reps 2800] [--label tag]
"""
import argparse
import asyncio
import json
import statistics as st
import time

import aiohttp

MODEL = "BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4-NextN-Graft"
BASE_TEXT = ("The development of large-scale GPU inference systems requires "
             "careful attention to memory bandwidth, kernel scheduling, and "
             "interconnect topology across the accelerator fleet. ")


async def one_request(session, url, prompt, osl, results, sem, stream):
    async with sem:
        body = {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": osl,
            "stream": stream,
            "ignore_eos": True,
            "temperature": 0.0,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        t0 = time.perf_counter()
        t_first = None
        t_last = None
        usage = None
        nchunks = 0
        try:
            async with session.post(url, json=body) as resp:
                if stream:
                    async for raw in resp.content:
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        now = time.perf_counter()
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        choices = obj.get("choices") or []
                        if choices and (choices[0].get("text") or ""):
                            nchunks += 1
                            if t_first is None:
                                t_first = now
                            t_last = now
                else:
                    text = await resp.text()
                    t_last = time.perf_counter()
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        results.append({"error": f"bad json: {text[:120]}"})
                        return
                    if resp.status >= 400:
                        results.append({"error": f"http {resp.status}: {text[:120]}"})
                        return
                    usage = obj.get("usage")
                    choices = obj.get("choices") or []
                    if choices and (choices[0].get("text") or ""):
                        nchunks += 1
                        t_first = t_last
        except Exception as exc:  # noqa: BLE001 — record and continue the sweep
            results.append({"error": f"{type(exc).__name__}: {exc}"})
            return
        if t_first is None or usage is None:
            results.append({"error": "no tokens or no usage"})
            return
        ct = usage.get("completion_tokens", 0)
        pt = usage.get("prompt_tokens", 0)
        ttft = t_first - t0
        if stream:
            decode_dur = max(t_last - t_first, 1e-9)
            # steady rate: tokens delivered after the first chunk over that span
            est_first_toks = max(1, round(ct / max(nchunks, 1)))
            steady = (ct - est_first_toks) / decode_dur if nchunks > 1 else 0.0
        else:
            steady = ct / max(t_last - t0, 1e-9)
        results.append({
            "ttft_ms": ttft * 1e3,
            "steady_tok_s": steady,
            "completion_tokens": ct,
            "prompt_tokens": pt,
            "wall_s": time.perf_counter() - t0,
            "chunks": nchunks,
        })


async def sweep(url, concurrencies, osl, text_reps, label, num_requests=0, stream=True):
    prompt = BASE_TEXT * text_reps
    mode = "stream" if stream else "nonstream"
    latency_label = "TTFT" if stream else "LAT"
    print(f"# label={label} mode={mode} url={url} osl={osl} text_reps={text_reps}")
    print(f"# {'C':>3s} {'reqs':>4s} {'ok':>3s} {'ISL':>6s} "
          f"{latency_label + '_p50_ms':>11s} {latency_label + '_p95_ms':>11s} "
          f"{'user_tok_s_p50':>14s} {'agg_out_tok_s':>13s} {'wall_s':>7s}")
    for c in concurrencies:
        nreq = num_requests if num_requests > 0 else max(8, 2 * c)
        sem = asyncio.Semaphore(c)
        results = []
        timeout = aiohttp.ClientTimeout(total=1200)
        conn = aiohttp.TCPConnector(limit=c + 4)
        t0 = time.perf_counter()
        async with aiohttp.ClientSession(timeout=timeout,
                                         connector=conn) as session:
            await asyncio.gather(*[
                one_request(session, url, prompt, osl, results, sem, stream)
                for _ in range(nreq)
            ])
        wall = time.perf_counter() - t0
        ok = [r for r in results if "error" not in r]
        errs = [r for r in results if "error" in r]
        if errs:
            print(f"#   C={c}: {len(errs)} errors, first: {errs[0]['error'][:120]}")
        if not ok:
            continue
        ttfts = sorted(r["ttft_ms"] for r in ok)
        steadies = sorted(r["steady_tok_s"] for r in ok)
        total_out = sum(r["completion_tokens"] for r in ok)
        isl = round(st.mean(r["prompt_tokens"] for r in ok))
        p = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]
        print(f"  {c:3d} {nreq:4d} {len(ok):3d} {isl:6d} "
              f"{p(ttfts, 0.5):11.0f} {p(ttfts, 0.95):11.0f} "
              f"{p(steadies, 0.5):14.2f} {total_out / wall:13.1f} {wall:7.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://10.42.0.44:8000/v1/completions")
    ap.add_argument("--concurrencies", default="1,4,8,16,32")
    ap.add_argument("--osl", type=int, default=256)
    ap.add_argument("--isl-text-reps", type=int, default=160)
    ap.add_argument("--label", default="run")
    ap.add_argument("--num-requests", type=int, default=0,
                    help="fixed request count per concurrency (0 = max(8, 2*C))")
    ap.add_argument("--model", default=MODEL,
                    help="served-model-name to target (routes to a pool)")
    ap.add_argument("--no-stream", action="store_false", dest="stream",
                    help="use non-streaming completions; reports end-to-end tokens/user/sec")
    a = ap.parse_args()
    MODEL = a.model
    cs = [int(x) for x in a.concurrencies.split(",")]
    asyncio.run(sweep(a.url, cs, a.osl, a.isl_text_reps, a.label, a.num_requests,
                      a.stream))
