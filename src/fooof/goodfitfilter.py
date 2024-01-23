""" goodfitfilter.py

"""
# Package Header #
from ...header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
from collections.abc import Iterable
from typing import Any

# Third-Party Packages #
from baseobjects import BaseObject
from baseobjects.typing import AnyCallable

# Local Packages #


# Definitions #
# Classes #
class GoodFitFilterBank(BaseObject):
    """

    Class Attributes:

    Attributes:

    Args:

    """
    # Magic Methods #
    # Construction/Destruction
    def __init__(self, init: bool = True) -> None:

        # New Attributes #
        self.filters: dict = {}
        self.filter_stack: list = []

        # Object Construction #
        if init:
            self.construct()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.filter(*args, **kwargs)

    # Instance Methods #
    # Constructors/Destructors
    def construct(self, ) -> None:
        pass

    def filter(self, *args: Any, **kwargs: Any) -> Any:
        """Cascade runs input through the stack of filters in order.

        Args:
            *args: The arguments input for the first filter.
            **kwargs: The keyword arguments for the first filter

        Returns:
            The results of the cascaded filter stack.
        """
        # Return the input if there are no filters.
        if not self.filter_stack:
            return args, kwargs

        # Cascade run the filters
        stack = self.filter_stack.copy()
        results = stack.pop(0)(*args, **kwargs)
        for filter_ in stack:
            try:
                results = filter_(*results)
            except TypeError:
                results = filter_(results)
        return results

    # Filters
    def clear_filters(self) -> None:
        """Removes all filters stored in this bank."""
        self.filters.clear()

    def add_filter(self, name: str, filter_: AnyCallable) -> None:
        """Add this filter to the bank.

        Args:
            name: The name of the filter.
            filter_: The filter store in this bank.
        """
        self.filters[name] = filter_
        self.filter_stack.append(filter_)

    def remove_filter(self, name: str) -> None:
        """Removes a filter from the bank.

        Args:
            name: The name of the filter to remove from the bank.
        """
        del self.filters[name]

    # Filter Stack
    def clear_stack(self) -> None:
        """Removes all filters in the stack."""
        self.filter_stack.clear()

    def add_filter_to_stack(self, name: str, filter_: AnyCallable | None = None) -> None:
        """Adds a filter to the stack which can be in this bank or a new filter to add.

        Args:
            name: The name of the filter to either get from the bank or to add if a filter is given.
            filter_: The filter put into the stack and store in this bank.
        """
        if filter_ is None:
            self.filter_stack.append(self.filters[name])
        else:
            self.filters[name] = filter_
            self.filter_stack.append(filter_)

    def set_filter(self, name: str, filter_: AnyCallable | None = None) -> None:
        """Sets the stack to have a single filter which can be in this bank or a new filter to add.

        Args:
            name: The name of the filter to either get from the bank or to add if a filter is given.
            filter_: The filter put into the stack and store in this bank.
        """
        self.clear_stack()
        self.add_filter_to_stack(name=name, filter_=filter_)

    def add_filters_to_stack(self, *args: str | Iterable[str | Iterable[str, AnyCallable]]):
        """Adds filters to the stack which can be in this bank or a new filter to add.

        Args:
            *args: An iterable of filters to add to the stack and to this bank.
        """
        if len(args) == 1:
            filters = args[0]
        else:
            filters = args

        for filter_ in filters:
            if isinstance(filter_, str):
                self.add_filter_to_stack(filter_)
            else:
                self.add_filter_to_stack(*filter_)


