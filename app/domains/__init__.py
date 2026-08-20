"""
Importing this package registers every available domain.

Adding a new domain later is: write app/domains/finance.py (mirroring
marketing.py's structure), then add one import line below. Nothing
else in the app changes.
"""

from app.domains import marketing  # noqa: F401  (import triggers registration)
