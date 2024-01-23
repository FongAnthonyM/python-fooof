""" goodnessauditinterface.py

"""
# Package Header #
from .....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import abc
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject

# Local Packages #


# Definitions #
# Classes #
class GoodnessAuditInterface(BaseObject):
    """An abstract class and interface for goodness audit objects."""
    # Magic Methods #
    # Callable
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """When this auditor is called, run the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self.run_audit(*args, **kwargs)

    # Instance Methods #
    # Auditing
    @abc.abstractmethod
    def run_audit(self, *args, **kwargs) -> Any:
        """Runs a goodness audit on the provided input.

        Args:
            *args: The arguments to run the audit.
            **kwargs: The keyword arguments to run the audit

        Returns:
            The results of the audit.
        """
        pass
