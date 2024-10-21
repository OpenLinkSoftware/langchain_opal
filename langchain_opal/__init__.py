from importlib import metadata

from langchain_opal.llms import OpalLLM
from langchain_opal.llms_assistant import OpalAssistantLLM
from langchain_opal.chat_models import ChatOpal
from langchain_opal.chat_models_assistant import ChatOpalAssistant

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = "0.1.1"
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "OpalLLM",
    "OpalAssistantLLM",
    "ChatOpal",
    "ChatOpalAssistant",
    "__version__",
]
