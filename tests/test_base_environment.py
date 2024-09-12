import pytest

from nsb.base import NSBEnvironment

# Create a dummy class that subclasses NSBEnvironment
class DummyEnvironment(NSBEnvironment):
    def __init__(self, public: int, private: int):
        self.public = public
        self._private = private

    @property
    def private(self) -> int:
        return self._private

    def step(self):
        pass

    def take_action(self, action: int):
        pass


def test_can_access_public_attr():
    env = DummyEnvironment(public=42, private=24)
    assert env.public == 42

def test_can_access_public_property():
    env = DummyEnvironment(public=42, private=24)
    # This is fine because the property is public,
    # even though the underlying attribute is private
    assert env.private == 24

def test_cannot_access_private_property_easily():
    env = DummyEnvironment(public=42, private=24)
    with pytest.raises(AttributeError):
        x = env._private
