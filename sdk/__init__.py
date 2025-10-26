"""
Cumulus SDK - Distributed Execution

A distributed execution system that sends code to remote GPU servers
with Cumulus GPU partitioning, similar to Modal but without Docker.
"""

from .client import CumulusClient
from .decorators import remote, gpu
from .code_packager import CodePackager

__version__ = "1.0.0"
__all__ = ["CumulusClient", "remote", "gpu", "CodePackager"]
