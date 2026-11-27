"""Bounded request/reply client compatible with OpenPI's metadata handshake."""

import os

from .wire import packb, unpackb, json_encode, json_decode


_DEFAULT_TIMEOUT = object()


class PolicyClient:
    def __init__(self, settings):
        self.settings = settings
        self.connection = None
        self.metadata = None

    def connect(self):
        from websockets.sync.client import connect

        key = os.environ.get(self.settings.get("api_key_env", "OPENPI_API_KEY"))
        headers = {"Authorization": f"Api-Key {key}"} if key else None
        self.connection = connect(self.settings["url"], compression=None,
            max_size=64 * 1024 * 1024, open_timeout=self.settings["timeout_s"],
            close_timeout=1, additional_headers=headers)
        try:
            payload = self.connection.recv(timeout=self.settings["timeout_s"])
            self.metadata = self._decode(payload)
            if not isinstance(self.metadata, dict):
                raise ValueError("policy metadata must be an object")
        except BaseException:
            self.close()
            raise
        return self.metadata

    def _decode(self, payload):
        if self.settings["protocol"] == "openpi":
            if isinstance(payload, str):
                raise RuntimeError(f"policy server error: {payload[:500]}")
            return unpackb(payload)
        return json_decode(payload)

    def infer(self, observation, *, timeout_s=_DEFAULT_TIMEOUT):
        """Request a prediction using the configured timeout unless explicitly overridden."""
        if self.connection is None:
            raise RuntimeError("policy is not connected")
        encode = packb if self.settings["protocol"] == "openpi" else json_encode
        self.connection.send(encode(observation))
        if timeout_s is _DEFAULT_TIMEOUT:
            timeout_s = self.settings["timeout_s"]
        response = self._decode(self.connection.recv(timeout=timeout_s))
        if not isinstance(response, dict):
            raise ValueError("policy response must be an object")
        return response

    def close(self):
        if self.connection is not None:
            connection = self.connection
            self.connection = None
            connection.close()
