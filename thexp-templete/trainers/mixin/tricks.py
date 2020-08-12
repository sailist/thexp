from thexp.decorators import clswrap


class TrickMixin():
    any_ = clswrap(lambda: 'function added here needs to wrap clswrap')
