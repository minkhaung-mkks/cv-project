#!/usr/bin/env python3
"""Run preprocessing, mask review, camera recovery, and training."""

import preprocess
import train


def main() -> None:
    preprocess.main()
    train.main()


if __name__ == "__main__":
    main()
