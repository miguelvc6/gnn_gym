import hashlib
import json
from collections.abc import Mapping


def config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]
