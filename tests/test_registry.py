"""Tests for FlashFusion Registry class."""

import pytest

from flashfusion.registry import Registry, BACKBONES, STRATEGIES, PIPELINES


class TestRegistry:
    """Test suite for the Registry pattern implementation."""

    def test_register_and_build(self):
        """Test registering a class and building it."""
        reg = Registry("test_components")

        @reg.register("MyComponent")
        class MyComponent:
            def __init__(self, value=42):
                self.value = value

        instance = reg.build("MyComponent", value=99)
        assert instance.value == 99

    def test_register_without_name(self):
        """Test registering with class name as default key."""
        reg = Registry("test")

        @reg.register()
        class AutoNamed:
            pass

        assert "AutoNamed" in reg
        instance = reg.build("AutoNamed")
        assert isinstance(instance, AutoNamed)

    def test_register_callable_directly(self):
        """Test registering a callable without parentheses."""
        reg = Registry("test")

        @reg.register
        class DirectRegister:
            pass

        assert "DirectRegister" in reg

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises KeyError."""
        reg = Registry("test")

        @reg.register("Duplicate")
        class First:
            pass

        with pytest.raises(KeyError, match="already registered"):
            @reg.register("Duplicate")
            class Second:
                pass

    def test_build_missing_raises(self):
        """Test that building unregistered name raises KeyError."""
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.build("NonExistent")

    def test_get(self):
        """Test getting a registered class without instantiation."""
        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            pass

        cls = reg.get("Foo")
        assert cls is Foo

    def test_get_missing_raises(self):
        """Test that getting unregistered name raises KeyError."""
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("Missing")

    def test_list(self):
        """Test listing all registered names."""
        reg = Registry("test")

        @reg.register("B")
        class B:
            pass

        @reg.register("A")
        class A:
            pass

        names = reg.list()
        assert names == ["A", "B"]

    def test_contains(self):
        """Test __contains__ (in operator)."""
        reg = Registry("test")

        @reg.register("Present")
        class Present:
            pass

        assert "Present" in reg
        assert "Absent" not in reg

    def test_len(self):
        """Test __len__ returns correct count."""
        reg = Registry("test")
        assert len(reg) == 0

        @reg.register("One")
        class One:
            pass

        assert len(reg) == 1

    def test_repr(self):
        """Test string representation."""
        reg = Registry("my_registry")
        repr_str = repr(reg)
        assert "my_registry" in repr_str

    def test_name_property(self):
        """Test name property."""
        reg = Registry("components")
        assert reg.name == "components"


class TestGlobalRegistries:
    """Test that global registry instances exist and are functional."""

    def test_backbones_registry_exists(self):
        """Test BACKBONES registry is a valid Registry."""
        assert isinstance(BACKBONES, Registry)
        assert BACKBONES.name == "backbones"

    def test_strategies_registry_exists(self):
        """Test STRATEGIES registry is a valid Registry."""
        assert isinstance(STRATEGIES, Registry)
        assert STRATEGIES.name == "strategies"

    def test_pipelines_registry_exists(self):
        """Test PIPELINES registry is a valid Registry."""
        assert isinstance(PIPELINES, Registry)
        assert PIPELINES.name == "pipelines"

    def test_strategies_has_wbf(self):
        """Test that WBF is registered in STRATEGIES."""
        assert "weighted_box_fusion" in STRATEGIES
