from thexp.decorators import clswrap


class TrickMixin():
    any_ = clswrap(lambda: print('function added here needs to wrap clswrap'))
