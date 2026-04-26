# train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['A', 'B', 'both'], default='both')
    args = parser.parse_args()

    if args.model in ('A', 'both'):
        print("\n=== Training Model A (TerraMind / HLS) ===")
        from src.model_a import train
        train()

    if args.model in ('B', 'both'):
        print("\n=== Training Model B (WildfireCNN / NDWS) ===")
        from src.model_b import train
        train()

if __name__ == "__main__":
    main()