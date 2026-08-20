#!/usr/bin/env python
"""
Run the RAG ingestion pipeline for one domain.

Usage:
    python scripts/ingest.py --domain marketing
    python scripts/ingest.py --domain marketing --reset false
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.domains  # noqa: F401 -- registers domains
from app.agent.domain_config import get_domain, list_domains
from app.rag.ingestion import ingest_domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a domain's knowledge base into the vector store.")
    parser.add_argument("--domain", required=True, help=f"One of: {list_domains()}")
    parser.add_argument("--reset", default="true", choices=["true", "false"])
    args = parser.parse_args()

    domain_config = get_domain(args.domain)
    if domain_config is None:
        print(f"Unknown domain '{args.domain}'. Available: {list_domains()}")
        raise SystemExit(1)

    count = ingest_domain(
        domain=domain_config.name,
        kb_path=domain_config.knowledge_base_path,
        reset=(args.reset == "true"),
    )
    print(f"Ingested {count} chunks for domain '{domain_config.name}' "
          f"from {domain_config.knowledge_base_path}")


if __name__ == "__main__":
    main()
