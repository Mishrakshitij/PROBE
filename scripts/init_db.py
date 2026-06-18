#!/usr/bin/env python3
"""Initialize and seed the flight database.

Usage:
    python scripts/init_db.py [--seed 42] [--db-path data/flights.db]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapo.environment.database import FlightDB
from aapo.environment.seed import DataGenerator, GeneratorConfig


def main():
    parser = argparse.ArgumentParser(description="Initialize flight database")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-path", type=str, default="data/flights.db")
    parser.add_argument("--num-airports", type=int, default=35)
    parser.add_argument("--num-routes", type=int, default=120)
    args = parser.parse_args()

    config = GeneratorConfig(
        num_airports=args.num_airports,
        num_routes=args.num_routes,
    )

    # Remove existing DB
    db_path = Path(args.db_path)
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing {db_path}")

    db = FlightDB(args.db_path)
    gen = DataGenerator(seed=args.seed, config=config)
    gen.populate(db)

    # Print some stats
    stats = {
        "airlines": db.query_one("SELECT count(*) as n FROM airlines")["n"],
        "airports": db.query_one("SELECT count(*) as n FROM airports")["n"],
        "routes": db.query_one("SELECT count(*) as n FROM routes")["n"],
        "flights": db.query_one("SELECT count(*) as n FROM flights")["n"],
        "fares": db.query_one("SELECT count(*) as n FROM fare_classes")["n"],
        "restricted": db.query_one("SELECT count(*) as n FROM routes WHERE is_restricted=1")["n"],
        "cancelled": db.query_one("SELECT count(*) as n FROM flights WHERE status='cancelled'")["n"],
        "promo_fares": db.query_one("SELECT count(*) as n FROM fare_classes WHERE fare_class='promo'")["n"],
    }
    print(f"\nDatabase stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    db.close()
    print(f"\nDatabase saved to {args.db_path}")


if __name__ == "__main__":
    main()
