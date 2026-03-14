#!/usr/bin/env python3
"""
Founder Research Agent – CLI entry point.

Usage:
    python main.py "Elon Musk" --context "CEO of Tesla and SpaceX" --iterations 4
    python main.py "Sam Altman" --output-dir ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("output/agent.log", mode="a"),
        ],
    )


def check_env() -> None:
    if not os.getenv("GROQ_API_KEY"):
        print("❌  GROK_API_KEY is not set. Please set it in your environment or .env file.")
        sys.exit(1)
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️   TAVILY_API_KEY not set – will use DuckDuckGo as fallback search engine.")


def run_agent(
    target_name: str,
    target_context: str = "",
    max_iterations: int = 4,
    output_dir: str = "output",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    from agent.graph import compile_graph

    graph = compile_graph()

    initial_state = {
        "target_name":    target_name,
        "target_context": target_context,
        "max_iterations": max_iterations,
    }

    print(f"\n🔍  Starting research on: {target_name}")
    print(f"⚙️   Max iterations: {max_iterations}")
    print("─" * 60)

    final_state = graph.invoke(initial_state)

    print("\n" + "─" * 60)
    print(f"✅  Research complete!")
    print(f"📄  Report saved to: {final_state.get('report_path', 'output/')}")
    print(f"🧠  Facts extracted: {len(final_state.get('extracted_facts', []))}")
    print(f"🌐  Pages scraped:   {len(final_state.get('scraped_pages', []))}")

    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Founder/CEO Research Agent powered by LangGraph"
    )
    parser.add_argument("name",            help="Full name of the founder/CEO to research")
    parser.add_argument("--context",       default="",    help="Optional context (e.g. 'CEO of OpenAI')")
    parser.add_argument("--iterations",    default=4,     type=int, help="Max research iterations (default: 4)")
    parser.add_argument("--output-dir",    default="output", help="Directory for reports (default: output/)")
    parser.add_argument("--log-level",     default="INFO",   help="Logging level (default: INFO)")
    parser.add_argument("--print-report",  action="store_true", help="Print final report to stdout")
    args = parser.parse_args()

    # Ensure output dir exists before logging setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    setup_logging(args.log_level)
    check_env()

    state = run_agent(
        target_name=args.name,
        target_context=args.context,
        max_iterations=args.iterations,
        output_dir=args.output_dir,
    )

    if args.print_report:
        print("\n" + "=" * 60)
        print(state.get("report_markdown", ""))

    sys.exit(0)


if __name__ == "__main__":
    main()
