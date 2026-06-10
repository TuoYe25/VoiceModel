"""
CLI runner for local benchmarking without the web server.

Usage:
    python run_benchmark.py --audio test.wav
    python run_benchmark.py --audio test.wav --models small medium turbo
    python run_benchmark.py --audio test.wav --ground-truth "expected text here"
"""

import argparse
import sys
import os

# Ensure backend is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models.base_engine import EngineConfig
from backend.models.registry import list_engines
from backend.benchmark import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Edge STT Model Benchmark Tool"
    )
    parser.add_argument(
        "--audio", required=True, help="Path to an audio file (WAV/FLAC/MP3)"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["small", "medium", "turbo"],
        help="Model IDs to benchmark (default: small medium turbo)"
    )
    parser.add_argument(
        "--engine", default="faster-whisper",
        help="Engine type (default: faster-whisper)"
    )
    parser.add_argument(
        "--compute-type", default="int8",
        choices=["float16", "int8", "int8_float16"],
        help="Compute precision (default: int8 for speed)"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--ground-truth", default=None,
        help="Reference text for WER calculation"
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip warmup inference"
    )
    parser.add_argument(
        "--output", default="./benchmark_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate audio
    if not os.path.isfile(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    # Print available models
    print("\n══════════════════════════════════════════════")
    print("  Edge STT Model Benchmark Tool")
    print("══════════════════════════════════════════════")

    engines = list_engines()
    for eng_name, models in engines.items():
        print(f"\n[{eng_name}] Available models:")
        for m in models:
            print(f"  • {m.model_id:20s} {m.param_count:>6s} params | {m.vram_requirement:>10s} VRAM | {m.description}")
    
    print(f"\n══════════════════════════════════════════════")
    print(f"  Audio: {args.audio}")
    print(f"  Models to test: {', '.join(args.models)}")
    print(f"  Engine: {args.engine} | Device: {args.device} | Compute: {args.compute_type}")
    print(f"══════════════════════════════════════════════\n")

    # Build configs
    configs = [
        EngineConfig(
            engine_type=args.engine,
            model_id=mid,
            device=args.device,
            compute_type=args.compute_type,
        )
        for mid in args.models
    ]

    # Run
    runner = BenchmarkRunner(output_dir=args.output)
    results = runner.run_benchmark(
        model_configs=configs,
        audio_files=[args.audio],
        ground_truths={args.audio: args.ground_truth} if args.ground_truth else None,
        warmup=not args.no_warmup,
    )

    # Report
    report = runner.generate_report(results)
    print("\n" + report)

    # Save
    json_path = runner.save_results(results)
    print(f"\nResults saved to: {json_path}")

    # Generate markdown
    md_path = os.path.join(args.output, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {md_path}")


if __name__ == "__main__":
    main()
