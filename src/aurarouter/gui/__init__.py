def check_pyside6() -> None:
    """Verify that PySide6 is available, raising a helpful error if not."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        raise ImportError(
            "PySide6 is required for the AuraRouter GUI.\n"
            "Install it with:  pip install PySide6>=6.6"
        ) from None
