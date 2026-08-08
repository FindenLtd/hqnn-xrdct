from __future__ import annotations

import argparse

from hybrid_qnn_xrdct.config import load_config
from hybrid_qnn_xrdct.data import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physically recalculated SOFC wt% mixtures")
    parser.add_argument("--config", default="configs/sofc.yaml")
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", default="data/sofc_200k.h5")
    args = parser.parse_args()
    output = generate_dataset(args.library, args.output, load_config(args.config))
    print(output.resolve())


if __name__ == "__main__":
    main()
