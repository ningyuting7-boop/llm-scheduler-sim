"""How fast is the predictor on its own, with nothing else on the GPU?

The benchmark measured 50ms per prediction batch while sharing an A100 with
Qwen3-8B. Subtracting the oracle arm, which runs the same code path without
the network, puts the DeBERTa forward at about 47.8ms of that. For a 184M
model at batch 1 that is roughly ten times what the hardware should need, so
the gap is presumed to be kernel queueing behind the served model's decode
steps -- but presumed is not measured, and this script measures it.

Run it on an idle GPU and compare against the 47.8ms. The difference is what
co-locating the predictor costs it. That number does not change scheduling
quality, since prediction is off the critical path, but it does say how much
headroom a separate GPU would buy if the arrival rate ever outgrew what one
shared GPU can score.

Reports per-batch and per-request latency across batch sizes, so the point
where batching starts to pay is visible. The benchmark's batching rule
(flush after 3ms) meant batches stayed small, so the per-request column is
the one that matched production there.

Usage:
    python scripts/bench_predictor_latency.py
    python scripts/bench_predictor_latency.py --batch-sizes 1 8 32 --seq-len 100
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--seq-len",
        type=int,
        default=100,
        help="tokens per prompt; the benchmark's mean input was 102",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--fp32", action="store_true", help="skip the fp16 cast")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from src.predictor_model import DEBERTA_MODEL_NAME, LengthDistributionPredictor

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU; the point is to measure it without contention")

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {name}")
    print(f"free {free / 2**30:.1f} GiB of {total / 2**30:.1f} GiB")
    if (total - free) / total > 0.1:
        print("WARNING: something else is already using this GPU; the result will\n"
              "         include contention, which is the thing being measured away")

    model = LengthDistributionPredictor.from_pretrained(DEBERTA_MODEL_NAME).eval()
    if not args.fp32:
        model = model.half()
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_NAME)

    n_params = sum(p.numel() for p in model.parameters())
    dtype = next(model.parameters()).dtype
    print(f"model: {n_params:,} params, {dtype}, seq_len={args.seq_len}\n")

    # One prompt repeated: this measures the forward, not tokenizer variance.
    prompt = " ".join(["token"] * args.seq_len)

    print(f"{'batch':>6}{'per batch':>12}{'per request':>13}{'p50':>9}{'p99':>9}{'req/s':>10}")
    print("-" * 59)
    for batch_size in args.batch_sizes:
        encoding = tokenizer(
            [prompt] * batch_size,
            max_length=args.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            for _ in range(args.warmup):
                model(input_ids, attention_mask)
            torch.cuda.synchronize()

            samples = []
            for _ in range(args.iters):
                start = time.perf_counter()
                model(input_ids, attention_mask)
                torch.cuda.synchronize()  # the forward is async without this
                samples.append((time.perf_counter() - start) * 1000)

        samples.sort()
        mean = statistics.fmean(samples)
        p50 = samples[len(samples) // 2]
        p99 = samples[min(len(samples) - 1, int(0.99 * len(samples)))]
        print(
            f"{batch_size:>6}{mean:>10.2f}ms{mean / batch_size:>11.2f}ms"
            f"{p50:>7.2f}ms{p99:>7.2f}ms{1000 * batch_size / mean:>10.0f}"
        )

    print(
        f"\nMeasured under contention with Qwen3-8B: ~47.8 ms per batch"
        f"\n(50.1 ms total minus the 2.3 ms the oracle arm spends on decode,"
        f"\ntemplate stripping and Monte Carlo without running the network)."
    )


if __name__ == "__main__":
    main()
