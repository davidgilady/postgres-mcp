from dataclasses import dataclass
from enum import Enum


class AccessMode(str, Enum):
    """SQL access modes for the server."""

    UNRESTRICTED = "unrestricted"  # Unrestricted access
    RESTRICTED = "restricted"  # Read-only with safety features


@dataclass(frozen=True)
class HostConfig:
    """Connection details for a single database host."""

    host: str
    port: int
    username: str
    password: str
