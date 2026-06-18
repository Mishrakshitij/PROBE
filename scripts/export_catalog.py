#!/usr/bin/env python3
"""Export failure catalog from collected episodes.

Usage:
    python scripts/export_catalog.py [--log-dir data/logs] [--output data/failure_catalog.json]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapo.replay.embedder import TrajectoryEmbedder
from aapo.replay.cluster import ClusterManager
from aapo.replay.catalog import FailureCatalog
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_episodes_from_logs


def main():
    parser = argparse.ArgumentParser(description="Export failure catalog")
    parser.add_argument("--log-dir", type=str, default="data/logs")
    parser.add_argument("--output", type=str, default="data/failure_catalog.json")
    args = parser.parse_args()

    episodes = load_episodes_from_logs(args.log_dir)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes found.")
        return

    embedder = TrajectoryEmbedder()
    cluster_mgr = ClusterManager(embedder)
    cluster_mgr.add_and_cluster(episodes)

    catalog = FailureCatalog(cluster_mgr)
    entries = catalog.export(episodes, args.output)
    print(f"Exported {len(entries)} failure clusters to {args.output}")

    for entry in entries[:5]:
        print(f"  Cluster {entry['cluster_id']}: {entry['size']} episodes, "
              f"policies={entry['violated_policies']}, severity={entry['max_severity']}")


if __name__ == "__main__":
    main()
