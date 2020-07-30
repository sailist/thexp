"""
A class for storing a tree graph. Primarily used for filter constructs in the
ORM.
"""
from typing import overload, Iterator
from datetime import datetime

import copy

from thexp.base_classes.qnode import Node


def is_iterable(x):
    """
    Copied from django.utils.itercompat.is_iterable
    -----

    An implementation independent way of checking for iterables
    """
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


def make_hashable(value):
    """copied from django.utils.hashable.make_hashable"""
    if isinstance(value, dict):
        return tuple([
            (key, make_hashable(nested_value))
            for key, nested_value in value.items()
        ])
    # Try hash to avoid converting a hashable iterable (e.g. string, frozenset)
    # to a tuple.
    try:
        hash(value)
    except TypeError:
        if is_iterable(value):
            return tuple(map(make_hashable, value))
        # Non-hashable, non-iterable.
        raise
    return value


class Q(Node):
    """
    Copied from django.db.models.query_utils.Q
    ------

    Encapsulate filters as objects that can then be combined logically (using
    `&` and `|`).
    """
    # Connection types
    AND = 'AND'
    OR = 'OR'
    default = AND
    conditional = True

    @overload
    def __init__(self,
                 test_name: str = None,
                 start_time: datetime = None,
                 end_time: datetime = None,
                 tags: Iterator = None,
                 plugins: Iterator = None,
                 projects: Iterator = None,
                 expnames: Iterator = None,
                 end_states: Iterator[int] = None,
                 states: Iterator[str] = None,
                 has__board: bool = None,
                 has__log: bool = None,
                 *args, **kwargs):
        """

        Args:
            test_name:
            start_time:
            end_time:
            tags:
            plugins:
            projects:
            expnames:
            end_states:
            states:
            has__board:
            has__log:
            *args:
            **kwargs:
        """

    def __init__(self, *args, _connector=None, _negated=False, **kwargs):
        super().__init__(children=[*args, *sorted(kwargs.items())], connector=_connector, negated=_negated)

    def _combine(self, other, conn):
        if not isinstance(other, Q):
            raise TypeError(other)

        # If the other Q() is empty, ignore it and just use `self`.
        if not other:
            return copy.deepcopy(self)
        # Or if this Q is empty, ignore it and just use `other`.
        elif not self:
            return copy.deepcopy(other)

        obj = type(self)()
        obj.connector = conn
        obj.add(self, conn)
        obj.add(other, conn)
        return obj

    def __or__(self, other):
        return self._combine(other, self.OR)

    def __and__(self, other):
        return self._combine(other, self.AND)

    def __invert__(self):
        obj = type(self)()
        obj.add(self, self.AND)
        obj.negate()
        return obj

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        # We must promote any new joins to left outer joins so that when Q is
        # used as an expression, rows aren't filtered due to joins.
        clause, joins = query._add_q(
            self, reuse, allow_joins=allow_joins, split_subq=False,
            check_filterable=False,
        )
        query.promote_joins(joins)
        return clause

    def deconstruct(self):
        args, kwargs = (), {}
        if len(self.children) == 1 and not isinstance(self.children[0], Q):
            child = self.children[0]
            kwargs = {child[0]: child[1]}
        else:
            args = tuple(self.children)
            if self.connector != self.default:
                kwargs = {'_connector': self.connector}
        if self.negated:
            kwargs['_negated'] = True
        return args, kwargs
