# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading


class KeyboardListener:
    def __init__(self):
        from pynput import keyboard

        self.state_lock = threading.Lock()
        self.latest_data = {"key": None}
        self._pressed_keys: set[str] = set()

        self.listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.listener.start()
        self.last_intervene = 0

    def on_key_press(self, key):
        key_str = key.char if hasattr(key, "char") else str(key)
        with self.state_lock:
            self.latest_data["key"] = key_str
            self._pressed_keys.add(key_str)

    def on_key_release(self, key):
        key_str = key.char if hasattr(key, "char") else str(key)
        with self.state_lock:
            self.latest_data["key"] = None
            self._pressed_keys.discard(key_str)

    def get_key(self) -> str | None:
        """Returns the currently held key."""
        with self.state_lock:
            return self.latest_data["key"]

    def consume_press(self, key: str) -> bool:
        """Returns True if key was pressed since last call, then clears it.

        Use this instead of get_key() for toggle keys to avoid missing
        short presses between polling cycles.
        """
        with self.state_lock:
            if key in self._pressed_keys:
                self._pressed_keys.discard(key)
                return True
            return False
