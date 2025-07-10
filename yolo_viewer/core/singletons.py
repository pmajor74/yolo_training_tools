"""Module-level singleton instances for the application.

This approach avoids all the metaclass and __new__ issues with PyQt6 and Python 3.13
by creating singleton instances at module level.
"""

# These will be initialized when first accessed
_model_cache = None
_settings_manager = None
_image_cache = None
_dataset_manager = None


def get_model_cache():
    """Get the singleton ModelCache instance."""
    global _model_cache
    if _model_cache is None:
        from .model_cache import ModelCache
        _model_cache = ModelCache()
    return _model_cache


def get_settings_manager():
    """Get the singleton SettingsManager instance."""
    global _settings_manager
    if _settings_manager is None:
        from .settings_manager import SettingsManager
        _settings_manager = SettingsManager()
    return _settings_manager


def get_image_cache():
    """Get the singleton ImageCache instance."""
    global _image_cache
    if _image_cache is None:
        from .image_cache import ImageCache
        _image_cache = ImageCache()
    return _image_cache


def get_dataset_manager():
    """Get the singleton DatasetManager instance."""
    global _dataset_manager
    if _dataset_manager is None:
        from .dataset_manager import DatasetManager
        _dataset_manager = DatasetManager()
    return _dataset_manager