"""Publish complete remote artifacts without changing existing local results."""
from pathlib import Path


def fetch_if_present(sftp, source, destination):
    """Missing or interrupted downloads leave no placeholder or partial result."""
    destination = Path(destination)
    try:
        sftp.stat(source)
    except (FileNotFoundError, PermissionError):
        return False
    temporary = destination.with_name(destination.name + ".download")
    try:
        sftp.get(source, str(temporary))
        temporary.replace(destination)
        return True
    except (FileNotFoundError, PermissionError):
        return False
    finally:
        temporary.unlink(missing_ok=True)
