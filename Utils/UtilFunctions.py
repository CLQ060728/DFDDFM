# Author: Qian Liu
# Email: liu.qian.pro@gmail.com

class ConfigDict(dict):
    """
    A dictionary that allows both dot notation and bracket access.
    Fully compatible with normal dict behavior.
    """
    def __getattr__(self, name):
        try:
            value = self[name]
            # Recursively wrap nested dicts
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{name}'")
