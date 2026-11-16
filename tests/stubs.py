"""
Empty stand-ins for camb / healpy / pymaster / iminuit, so that modules which import
them at the top can be imported (and their pure-python logic tested) on a machine
without those packages. Any real call into a stub raises AttributeError.
"""
import sys
import types


def install():
    for name in ("camb", "healpy", "pymaster", "iminuit"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            if name == "iminuit":
                mod.Minuit = object
            sys.modules[name] = mod
