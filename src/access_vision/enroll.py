from __future__ import annotations

import argparse
import logging

from .config import load_config
from .server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll face folders through a localhost browser")
    parser.add_argument("--config", default="config.toml", help="TOML configuration path")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run_server(load_config(args.config), enrollment_only=True)


if __name__ == "__main__":
    main()
