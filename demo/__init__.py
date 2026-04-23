def main(*args, **kwargs):
    from .demo import main as _main

    return _main(*args, **kwargs)


__all__ = ["main"]
