""" goodnessauditor.py

"""
# Package Header #
from ....header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import MutableMapping, Hashable, Iterator, Iterable
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
from baseobjects.typing import AnyCallable

# Local Packages #
from .audits import GoodnessAuditInterface


# Definitions #
# Classes #
class GoodnessAuditor(BaseObject, MutableMapping):
    """An object which evaluates the goodness of a fit curve based on a chosen audit.

    The Auditor contains several audits within it, so the method of determining goodness can be changed efficiently.

    Class Attributes:
        default_audits: The default audits to add when an object is created.

    Attributes:
        audits: The audits to store in this auditor.
        _audit: The audit to run when run_audit is called.

    Args:
        dict_: A dictionary of audits to add to this auditor.
        init: Determines if this object will be constructed.
        **kwargs: Audits to add as keyword arguments.
    """
    default_audits: dict[Hashable, GoodnessAuditInterface] = {}

    # Class Methods #
    @classmethod
    def fromkeys(cls, iterable: Iterable[Hashable], value: AnyCallable | None = None) -> "GoodnessAuditor":
        """Creates a GoodnessAuditor object with specified keys set to a default audit or None.

        Args:
            iterable: The keys put in audits.
            value: The default audit or None to fill the keys with.

        Returns:
            A GoodnessAuditor object with the keys.
        """
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    # Magic Methods #
    # Construction/Destruction
    def __init__(
        self,
        dict_: dict[Hashable, GoodnessAuditInterface] | None = None,
        /,
        init: bool = True,
        **kwargs: GoodnessAuditInterface,
    ) -> None:
        # New Attributes #
        self.audits: dict[Hashable, GoodnessAuditInterface] = self.default_audits.copy()
        self._audit: GoodnessAuditInterface | None = None

        # Object Construction #
        if init:
            self.construct(dict_, **kwargs)

    # Container Methods
    def __len__(self) -> int:
        """Returns the mount of audits in this auditor."""
        return len(self.audits)

    def __getitem__(self, key: Hashable) -> Any:
        """Gets an audit within this auditor."""
        if key in self.audits:
            return self.audits[key]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __setitem__(self, key: Hashable, item: Any) -> None:
        """Sets an audit within this auditor."""
        self.audits[key] = item

    def __delitem__(self, key: Hashable) -> Any:
        """Deletes and audit within this auditor."""
        del self.audits[key]

    def __iter__(self) -> Iterator:
        """Returns an iterator of the audits within this auditor."""
        return iter(self.audits)

    def __contains__(self, key: Hashable) -> bool:
        """Checks if an audit is within this auditor."""
        return key in self.audits

    # Callable
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """When this auditor is called, run the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self._audit(*args, **kwargs)

    # Representation
    def __repr__(self) -> str:
        """The representation of this object as a string."""
        return repr(self.audits)

    # Logic Operators
    def __or__(self, other: Any) -> "GoodnessAuditor":
        """The or operator as a union operation."""
        new = self.copy()
        if isinstance(other, GoodnessAuditor):
            return new.audits.update(other.audits)
        if isinstance(other, dict):
            return new.audits.update(other)
        return NotImplemented

    def __ror__(self, other: Any) -> "GoodnessAuditor":
        """The reverse or operator as a union operation."""
        new = self.copy()
        if isinstance(other, GoodnessAuditor):
            return new.audits.update(other.audits)
        if isinstance(other, dict):
            return new.audits.update(other)
        return NotImplemented

    def __ior__(self, other: Any) -> "GoodnessAuditor":
        """The assign or operator as a union operation"""
        if isinstance(other, GoodnessAuditor):
            self.audits |= other.audits
        else:
            self.audits |= other
        return self

    # Instance Methods #
    # Constructors/Destructors
    def construct(
        self,
        dict_: dict[Hashable, GoodnessAuditInterface] | None = None,
        /,
        **kwargs: GoodnessAuditInterface,
    ) -> None:
        """Constructs this object.

        Args:
            dict_: A dictionary of audits to add to this auditor.
            **kwargs: Audits to add as keyword arguments.
        """
        if dict_ is not None:
            self.update(dict)

        if kwargs:
            self.update(kwargs)

    # Auditing
    def run_audit(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the audit.

        Args:
            *args: The arguments to pass to the audit.
            **kwargs: The keyword arguments to pass to the audit.

        Returns:
            The results of the audit.
        """
        return self._audit(*args, **kwargs)

    def set_audit(self, name: Hashable, audit: GoodnessAuditInterface | None = None) -> None:
        """Sets the audit to be run. It can audit in this auditor or another to be added.

        Args:
            name: The name of the audit to set as the run audit.
            audit: The audit to add if given.
        """
        if audit is None:
            self._audit = self.audits[name]
        else:
            self.audits[name] = audit
            self._audit = audit
