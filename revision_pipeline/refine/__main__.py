"""Small synthetic acceptance run; this command does not load biological datasets."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Staged GenoDR development validation")
    parser.add_argument("command", choices=["smoke", "gpu-probe"])
    args = parser.parse_args()
    if args.command == "gpu-probe":
        from .smoke import gpu_probe
        return gpu_probe()
    from .runtime import configure_cpu
    runtime = configure_cpu()
    from .smoke import smoke
    path = smoke(runtime)
    print(f"Synthetic acceptance run saved: {path}")
    print("This is software validation, not a biological benchmark or reviewer experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
