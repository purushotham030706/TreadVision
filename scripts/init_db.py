"""
TreadVision: Database Initialization CLI
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_demo_assets import generate_all_demo_assets
from app.database.seed_data import seed_database


def main():
    print("==================================================")
    print("  TreadVision: Database & Demo Asset Initializer  ")
    print("==================================================")
    print("\nStep 1: Generating reference tread patterns and crime-scene impressions...")
    generate_all_demo_assets()

    print("\nStep 2: Initializing SQLite database and caching CV feature vectors...")
    seed_database(force_reseed=True)

    print("\nInitialization Complete! Database is populated and ready.")


if __name__ == "__main__":
    main()
