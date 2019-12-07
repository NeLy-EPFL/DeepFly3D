import logging


def error(msg, *args, **kwargs):
    getLogger().error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    getLogger().warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    getLogger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    getLogger().debug(msg, *args, **kwargs)


def info_enabled():
    return getLogger().isEnabledFor(logging.INFO)


def debug_enabled():
    return getLogger().isEnabledFor(logging.DEBUG)


def getLogger():
    return logging.getLogger(__name__)