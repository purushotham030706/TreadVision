"""
TreadVision: Database Initialization and Integrity CLI
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_demo_assets import generate_all_demo_assets
from app.database.connection import verify_database_integrity
from app.database.seed_data import seed_database


def main():
    print("==================================================")
    print("  TreadVision: Database & Demo Asset Initializer  ")
    print("==================================================")
    print("\n[1] Generating reference tread patterns and calibrated assets...")
    generate_all_demo_assets()

    print("\n[2] Initializing SQLite database and caching CV feature vectors...")
    seed_database(force_reseed=True)

    print("\n[3] Verifying database schema and foreign key integrity...")
    integrity = verify_database_integrity()
    if integrity["valid"]:
        print("    Database Integrity Check: PASSED")
        print(f"    Indexed Records: {integrity['counts']}")
    else:
        print("    Database Integrity Check: FAILED")
        if integrity["missing_tables"]:
            print(f"    Missing Tables: {integrity['missing_tables']}")
        if integrity["relationship_errors"]:
            print(f"    Relationship Errors: {integrity['relationship_errors']}")

    print("\nInitialization Complete! Database is populated and ready.")


if __name__ == "__main__":
    main()
