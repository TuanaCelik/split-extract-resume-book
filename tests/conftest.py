"""
Pytest configuration for extract-basic tests.

IMPORTANT: FAKE_LLAMA_CLOUD must be set before any test modules are imported,
as extraction_review.clients reads this at module load time to initialize
the mock server.
"""

import logging
import os
import sys

# Configure logging to stdout at INFO level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Enable the fake LlamaCloud server for all tests
os.environ["FAKE_LLAMA_CLOUD"] = "true"
