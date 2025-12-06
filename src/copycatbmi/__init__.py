from .bmi import CopyCat
from .troute import TRouteWarmer

from importlib.metadata import version as _version # This should cause it to be mangled, yes?

try:
    __version__ = _version("copycatbmi")  # Replace "my_package" with your actual package name
except Exception:
    # Fallback for local development or when the package is not yet installed
    __version__ = "unknown"