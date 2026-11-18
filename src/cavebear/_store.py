"""Small-file IO on local paths and cloud URLs, through obstore.

``storage_options`` is the dict that polars and deltalake accept: object_store
configuration keys, in any case. obstore rejects some keys that those libraries
accept, so they are dropped here, and client keys go to ``client_options``.
"""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import obstore
from obstore.exceptions import AlreadyExistsError
from obstore.store import ObjectStore, from_url

#: Keys that only polars or deltalake read. obstore raises on them.
IGNORED_KEYS = {
    "max_retries",
    "file_cache_ttl",
    "aws_s3_locking_provider",
    "delta_dynamo_table_name",
    "aws_s3_allow_unsafe_rename",
}
IGNORED_PREFIXES = ("retry_",)

#: object_store client keys, from obstore's ``ClientConfig``. obstore panics if
#: they are given in ``config``.
CLIENT_KEYS = {
    "allow_http",
    "allow_invalid_certificates",
    "connect_timeout",
    "default_content_type",
    "default_headers",
    "pool_idle_timeout",
    "pool_max_idle_per_host",
    "proxy_url",
    "proxy_ca_certificate",
    "proxy_excludes",
    "root_certificate",
    "randomize_addresses",
    "read_timeout",
    "timeout",
    "user_agent",
}


def _split_options(
    storage_options: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, client_options = {}, {}
    for key, value in (storage_options or {}).items():
        lower = key.lower()
        if lower in IGNORED_KEYS or lower.startswith(IGNORED_PREFIXES):
            continue
        if lower in CLIENT_KEYS:
            client_options[lower] = value
        else:
            config[key] = value
    return config, client_options


def open_store(
    root: str | Path, storage_options: dict[str, Any] | None = None, *, mkdir: bool = False
) -> ObjectStore:
    """An object store whose paths are relative to ``root``.

    For a local ``root`` that does not exist, this raises ``FileNotFoundError``,
    unless ``mkdir`` is true.
    """
    root = str(root)
    url = root if "://" in root else Path(root).absolute().as_uri()
    config, client_options = _split_options(storage_options)
    kwargs: dict[str, Any] = {}
    if config:
        kwargs["config"] = config
    if client_options:
        kwargs["client_options"] = client_options
    if url.startswith("file://"):
        local = Path(url2pathname(urlparse(url).path))
        if mkdir:
            kwargs["mkdir"] = True
        elif not local.is_dir():
            raise FileNotFoundError(f"no directory at {local}")
    return from_url(url, **kwargs)


def read_bytes(store: ObjectStore, path: str) -> bytes:
    """The bytes at ``path``. A missing file raises ``FileNotFoundError``."""
    return bytes(obstore.get(store, path).bytes())


def write_bytes(store: ObjectStore, path: str, data: bytes, *, overwrite: bool = False) -> None:
    """Write ``data`` to ``path``. Without ``overwrite``, an existing file raises
    ``FileExistsError``."""
    try:
        obstore.put(store, path, data, mode="overwrite" if overwrite else "create")
    except AlreadyExistsError as error:
        raise FileExistsError(f"{path} already exists") from error
