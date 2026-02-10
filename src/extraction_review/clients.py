import logging
import os

from llama_cloud import AsyncLlamaCloud

from .testing_utils import FakeLlamaCloudServer

logger = logging.getLogger(__name__)

# deployed agents may infer their name from the deployment name
# Note: Make sure that an agent deployment with this name actually exists
# otherwise calls to get or set data will fail. You may need to adjust the `or `
# name for development
agent_name = os.getenv("LLAMA_DEPLOY_DEPLOYMENT_NAME")
# required for all llama cloud calls
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
# get this in case running against a different environment than production
base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
project_id = os.getenv("LLAMA_DEPLOY_PROJECT_ID")

if os.getenv("FAKE_LLAMA_CLOUD"):
    fake: FakeLlamaCloudServer | None = FakeLlamaCloudServer().install()
else:
    fake = None


def get_llama_cloud_client() -> AsyncLlamaCloud:
    """Cloud services connection for file storage and processing."""
    return AsyncLlamaCloud(api_key=api_key, base_url=base_url)
