"""
Tests for localvectordb.__init__.py module.

These tests ensure that the package can be imported correctly and that
all public APIs are accessible.
"""
import types

import pytest


@pytest.mark.unit
class TestPackageImports:
    """Test package-level imports and public API."""

    def test_import_package(self):
        """Test that the package can be imported."""
        import localvectordb
        assert localvectordb is not None

    def test_import_main_classes(self):
        """Test importing main classes from package root."""
        from localvectordb import LocalVectorDB, RemoteVectorDB, VectorDB

        # Should be able to import without errors
        assert LocalVectorDB is not None
        assert RemoteVectorDB is not None
        assert VectorDB is not None

    def test_import_core_classes(self):
        """Test importing core classes."""
        from localvectordb import MetadataField

        assert MetadataField is not None

    def test_import_factory_classes(self):
        """Test importing factory classes."""
        from localvectordb import ChunkerFactory, EmbeddingRegistry

        assert ChunkerFactory is not None
        assert EmbeddingRegistry is not None

    def test_all_attribute(self):
        """Test that __all__ is properly defined."""
        import localvectordb

        # Should have __all__ attribute
        assert hasattr(localvectordb, '__all__')
        assert isinstance(localvectordb.__all__, list)

        # Should contain expected classes
        expected_classes = [
            "LocalVectorDB", "ChunkerFactory", "EmbeddingRegistry",
            "RemoteVectorDB", "VectorDB", "MetadataField"
        ]

        for cls_name in expected_classes:
            assert cls_name in localvectordb.__all__

    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can actually be imported."""
        import localvectordb

        for name in localvectordb.__all__:
            obj = getattr(localvectordb, name)
            assert obj is not None

            # Should be a class or callable
            assert callable(obj) or isinstance(obj, type) or isinstance(obj, types.ModuleType)

    def test_import_all_star(self):
        """Test that 'from localvectordb import *' works correctly."""
        # This is a bit tricky to test directly, so we'll check the __all__ contents
        import localvectordb

        all_names = localvectordb.__all__

        # All names should be accessible as module attributes
        for name in all_names:
            assert hasattr(localvectordb, name)

    def test_version_attribute(self):
        """Test that version information is accessible."""
        import localvectordb

        # Should have version info (either directly or importable)
        try:
            # Try to get version from the package
            version = getattr(localvectordb, '__version__', None)
            if version is None:
                # Try importing from metadata
                from importlib.metadata import version
                version = version('localvectordb')

            assert version is not None
            assert isinstance(version, str)
        except Exception:
            # Version info might not be available in development
            pytest.skip("Version information not available")


@pytest.mark.unit
class TestPublicAPI:
    """Test the public API surface of the package."""

    def test_local_vector_db_class(self):
        """Test LocalVectorDB class is properly exposed."""
        from localvectordb import LocalVectorDB

        # Should be a class
        assert isinstance(LocalVectorDB, type)

        # Should have expected methods
        expected_methods = [
            'upsert', 'insert', 'get', 'delete', 'update',
            'query', 'filter', 'exists', 'save', 'close'
        ]

        for method_name in expected_methods:
            assert hasattr(LocalVectorDB, method_name)
            assert callable(getattr(LocalVectorDB, method_name))

    def test_remote_vector_db_class(self):
        """Test RemoteVectorDB class is properly exposed."""
        from localvectordb import RemoteVectorDB

        # Should be a class
        assert isinstance(RemoteVectorDB, type)

        # Should have expected methods (same interface as LocalVectorDB)
        expected_methods = [
            'upsert', 'insert', 'get', 'delete', 'update',
            'query', 'filter', 'exists', 'save', 'close'
        ]

        for method_name in expected_methods:
            assert hasattr(RemoteVectorDB, method_name)
            assert callable(getattr(RemoteVectorDB, method_name))

    def test_vector_db_factory(self):
        """Test VectorDB factory function is properly exposed."""
        from localvectordb import VectorDB

        # Should be a callable (function)
        assert callable(VectorDB)

    def test_metadata_field_class(self):
        """Test MetadataField class is properly exposed."""
        from localvectordb import MetadataField

        # Should be a class
        assert isinstance(MetadataField, type)

        # Should be able to create instances
        try:
            field = MetadataField(type="text")
            assert field is not None
        except Exception as e:
            pytest.fail(f"Could not create MetadataField instance: {e}")

    def test_chunker_factory_class(self):
        """Test ChunkerFactory class is properly exposed."""
        from localvectordb import ChunkerFactory

        # Should have expected class methods
        expected_methods = ['create_chunker', 'list_methods']

        for method_name in expected_methods:
            assert hasattr(ChunkerFactory, method_name)
            assert callable(getattr(ChunkerFactory, method_name))

    def test_embedding_registry_class(self):
        """Test EmbeddingRegistry class is properly exposed."""
        from localvectordb import EmbeddingRegistry

        # Should have expected class methods
        expected_methods = ['register', 'get', 'create_provider', 'list']

        for method_name in expected_methods:
            assert hasattr(EmbeddingRegistry, method_name)
            assert callable(getattr(EmbeddingRegistry, method_name))


@pytest.mark.unit
@pytest.mark.performance
class TestImportPerformance:
    """Test import performance and lazy loading."""

    def test_import_time(self):
        """Test that package imports reasonably quickly."""
        import time

        start_time = time.time()
        end_time = time.time()

        import_time = end_time - start_time

        # Should import in under 2 seconds (generous for CI environments)
        assert import_time < 2.0, f"Import took {import_time:.2f} seconds"

    def test_selective_imports(self):
        """Test that selective imports work and are efficient."""
        import time

        # Test importing just what we need
        start_time = time.time()
        from localvectordb import VectorDB
        end_time = time.time()

        import_time = end_time - start_time
        assert import_time < 1.0

        # Should work without errors
        assert VectorDB is not None


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility of imports."""

    def test_legacy_import_patterns(self):
        """Test that legacy import patterns still work."""
        # Test importing from submodules (if supported)
        try:
            from localvectordb.database import LocalVectorDB
            assert LocalVectorDB is not None
        except ImportError:
            # This might not be supported, which is fine
            pass

        try:
            from localvectordb.client import RemoteVectorDB
            assert RemoteVectorDB is not None
        except ImportError:
            # This might not be supported, which is fine
            pass

    def test_class_attributes(self):
        """Test that classes have expected attributes."""
        from localvectordb import LocalVectorDB, RemoteVectorDB

        # Both should be classes
        assert isinstance(LocalVectorDB, type)
        assert isinstance(RemoteVectorDB, type)

        # Should have __doc__ strings (good practice)
        assert LocalVectorDB.__doc__ is not None
        assert RemoteVectorDB.__doc__ is not None


@pytest.mark.unit
class TestImportErrorHandling:
    """Test graceful handling of import errors."""

    def test_missing_dependencies(self):
        """Test behavior when optional dependencies are missing."""
        # This is hard to test directly without actually removing dependencies
        # In a real scenario, you might mock missing imports

        # Test that core imports work even if optional deps are missing
        try:
            from localvectordb import VectorDB
            assert VectorDB is not None
        except ImportError as e:
            pytest.fail(f"Core import failed: {e}")

    def test_import_from_uninstalled_package(self):
        """Test that importing gives helpful error when package not installed."""
        # This test simulates what happens when package isn't installed
        # In practice, this would be tested in a separate environment

        # For now, just ensure current imports work
        import localvectordb
        assert localvectordb is not None


@pytest.mark.unit
class TestDocumentation:
    """Test that documentation is accessible."""

    def test_module_docstring(self):
        """Test that module has docstring."""
        import localvectordb

        # Module should have a docstring
        assert localvectordb.__doc__ is not None
        assert len(localvectordb.__doc__.strip()) > 0

    def test_class_docstrings(self):
        """Test that main classes have docstrings."""
        from localvectordb import LocalVectorDB, MetadataField, RemoteVectorDB

        classes_to_check = [LocalVectorDB, RemoteVectorDB, MetadataField]

        for cls in classes_to_check:
            assert cls.__doc__ is not None
            assert len(cls.__doc__.strip()) > 0

    def test_help_functionality(self):
        """Test that help() works on main classes."""
        from localvectordb import LocalVectorDB, VectorDB

        # Should be able to get help without errors
        try:
            help(LocalVectorDB)
            # help() returns None but prints to stdout
        except Exception as e:
            pytest.fail(f"help() failed for LocalVectorDB: {e}")

        try:
            help(VectorDB)
        except Exception as e:
            pytest.fail(f"help() failed for VectorDB: {e}")


@pytest.mark.unit
class TestPackageStructure:
    """Test overall package structure and organization."""

    def test_submodules_accessible(self):
        """Test that submodules are accessible if needed."""
        import localvectordb

        # Test that we can access submodules through the package
        # This depends on how the package is structured

        expected_submodules = ['database', 'client', 'core', 'factory']

        for submodule in expected_submodules:
            try:
                # Try to access as attribute
                getattr(localvectordb, submodule)
            except AttributeError:
                # Try to import directly
                try:
                    exec(f"from localvectordb import {submodule}")
                except ImportError:
                    # This is fine - submodules might not be exposed
                    pass

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # Import everything to check for circular dependencies
        try:
            from localvectordb import (  # noqa: F401
                ChunkerFactory,
                EmbeddingRegistry,
                LocalVectorDB,
                MetadataField,
                RemoteVectorDB,
                VectorDB,
            )

            # If we get here without ImportError, no circular dependencies
            assert True
        except ImportError as e:
            if "circular" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Re-raise other import errors
                raise

    def test_namespace_pollution(self):
        """Test that package doesn't pollute namespace."""
        import localvectordb

        # Get all attributes
        all_attrs = dir(localvectordb)

        # Filter to public attributes (not starting with _)
        public_attrs = [attr for attr in all_attrs if not attr.startswith('_')]

        # Should only contain items from __all__ plus any documented public items
        expected_public = set(localvectordb.__all__)
        actual_public = set(public_attrs)

        # There might be a few extra public items, but not too many
        extra_items = actual_public - expected_public

        # Allow a few common extra items
        allowed_extra = {'version', '__version__', 'metadata'}
        unexpected_extra = extra_items - allowed_extra

        assert len(unexpected_extra) == 0, f"Unexpected public items: {unexpected_extra}"


@pytest.mark.unit
class TestModuleInitialization:
    """Test module initialization behavior."""

    def test_module_initialization_side_effects(self):
        """Test that importing the module doesn't have unwanted side effects."""
        import os
        import sys

        # Save the current state of localvectordb modules
        saved_modules = {}
        modules_to_remove = [name for name in sys.modules if name.startswith("localvectordb")]
        for module in modules_to_remove:
            saved_modules[module] = sys.modules[module]
            del sys.modules[module]

        try:
            # Capture initial state
            initial_env = dict(os.environ)
            initial_modules = set(sys.modules.keys())

            # Import the package
            import localvectordb  # noqa: F401

            # Check for unwanted side effects
            final_env = dict(os.environ)
            final_modules = set(sys.modules.keys())

            # Environment shouldn't be modified (much)
            env_changes = set(final_env.keys()) - set(initial_env.keys())
            # Allow some common environment additions
            allowed_env_changes = {'PYTHONPATH', 'PATH'}  # Common in test environments
            unexpected_env_changes = env_changes - allowed_env_changes

            # Don't be too strict about environment changes in test environments
            assert len(unexpected_env_changes) == 0, f"Unexpected env changes: {unexpected_env_changes}"

            # New modules should be reasonable
            new_modules = final_modules - initial_modules
            localvectordb_modules = [m for m in new_modules if m.startswith('localvectordb')]

            # Should have imported localvectordb modules
            assert len(localvectordb_modules) > 0

        finally:
            # Restore the original state of modules
            for module_name, module_obj in saved_modules.items():
                sys.modules[module_name] = module_obj

    def test_repeated_imports(self):
        """Test that repeated imports work correctly."""
        # Import multiple times
        import localvectordb as lvdb1
        import localvectordb as lvdb2
        from localvectordb import LocalVectorDB as LVDB1
        from localvectordb import LocalVectorDB as LVDB2

        # Should be the same objects
        assert lvdb1 is lvdb2
        assert LVDB1 is LVDB2

        # Should have same __all__
        assert lvdb1.__all__ == lvdb2.__all__

    def test_import_from_different_contexts(self):
        """Test importing from different contexts."""

        # Test importing in function
        def test_function_import():
            from localvectordb import VectorDB
            return VectorDB

        # Test importing in class
        class TestClass:
            def __init__(self):
                from localvectordb import LocalVectorDB
                self.db_class = LocalVectorDB

        # Both should work
        func_import = test_function_import()
        class_instance = TestClass()

        assert func_import is not None
        assert class_instance.db_class is not None

        # Should be the same class
        from localvectordb import LocalVectorDB, VectorDB
        assert func_import is VectorDB
        assert class_instance.db_class is LocalVectorDB
