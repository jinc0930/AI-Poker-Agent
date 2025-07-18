"""
Timeout decorator.

    :copyright: (c) 2012-2013 by PN.
    :license: MIT, see LICENSE for more details.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import sys
import time
import multiprocessing
import signal
from functools import wraps

############################################################
# Timeout
############################################################

# http://www.saltycrane.com/blog/2010/04/using-python-timeout-decorator-uploading-s3/
# Used work of Stephen "Zero" Chappell <Noctis.Skytower@gmail.com>
# in https://code.google.com/p/verse-quiz/source/browse/trunk/timeout.py


class TimeoutError(AssertionError):

    """Thrown when a timeout occurs in the `timeout` context manager."""

    def __init__(self, value="Timed Out"):
        self.value = value

    def __str__(self):
        return repr(self.value)


def _raise_exception(exception, exception_message):
    """ This function checks if a exception message is given.

    If there is no exception message, the default behaviour is maintained.
    If there is an exception message, the message is passed to the exception with the 'value' keyword.
    """
    if exception_message is None:
        raise exception()
    else:
        raise exception(exception_message)


def timeout(seconds=None, use_signals=True, timeout_exception=TimeoutError, exception_message=None):
    """Add a timeout parameter to a function and return it.

    :param seconds: optional time limit in seconds or fractions of a second. If None is passed, no timeout is applied.
        This adds some flexibility to the usage: you can disable timing out depending on the settings.
    :type seconds: float
    :param use_signals: flag indicating whether signals should be used for timing function out or the multiprocessing
        When using multiprocessing, timeout granularity is limited to 10ths of a second.
    :type use_signals: bool

    :raises: TimeoutError if time limit is reached

    It is illegal to pass anything other than a function as the first
    parameter. The function is wrapped and returned to the caller.
    """
    def decorate(function):

        if not seconds:
            return function

        if use_signals:
            def handler(signum, frame):
                _raise_exception(timeout_exception, exception_message)

            @wraps(function)
            def new_function(*args, **kwargs):
                new_seconds = kwargs.pop('timeout', seconds)
                if new_seconds:
                    old = signal.signal(signal.SIGALRM, handler)
                    signal.setitimer(signal.ITIMER_REAL, new_seconds)
                try:
                    return function(*args, **kwargs)
                finally:
                    if new_seconds:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, old)
            return new_function
        else:
            @wraps(function)
            def new_function(*args, **kwargs):
                timeout_wrapper = _Timeout(function, timeout_exception, exception_message, seconds)
                return timeout_wrapper(*args, **kwargs)
            return new_function

    return decorate

def timeout2(seconds=None, defaultretval="Blah",exception_message="[EXP]: Action TimedOut",timeout_exception=TimeoutError):
    """
        Similar as before return a default value instead.
        Uses Signals. Can you use multiprocessing instead.
    """
    def decorate(function):

        if not seconds:
            return function

        
        def handler(signum, frame):
            _raise_exception(timeout_exception, exception_message)
            print("[EXP] : TimedOut, Returning Default Value (Fold)")
            #print(defaultretval)
            #return defaultretval
        @wraps(function)
        def new_function(*args, **kwargs):
            new_seconds = kwargs.pop('timeout', seconds)
            if new_seconds:
                print("[EXP] : No-TimeOut")
                old = signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, new_seconds)
            try:
                return function(*args, **kwargs)
            except TimeoutError :
                print(exception_message)
                return defaultretval
            finally:
                if new_seconds:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
        return new_function

    return decorate

def _target(queue, function, *args, **kwargs):
    """Run a function with arguments and return output via a queue.

    This is a helper function for the Process created in _Timeout. It runs
    the function with positional arguments and keyword arguments and then
    returns the function's output by way of a queue. If an exception gets
    raised, it is returned to _Timeout to be raised by the value property.
    """
    try:
        queue.put((True, function(*args, **kwargs)))
    except:
        queue.put((False, sys.exc_info()[1]))


class _Timeout(object):

    """Wrap a function and add a timeout (limit) attribute to it.

    Instances of this class are automatically generated by the add_timeout
    function defined above. Wrapping a function allows asynchronous calls
    to be made and termination of execution after a timeout has passed.
    """

    def __init__(self, function, timeout_exception, exception_message, limit):
        """Initialize instance in preparation for being called."""
        self.__limit = limit
        self.__function = function
        self.__timeout_exception = timeout_exception
        self.__exception_message = exception_message
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.__timeout = time.time()
        self.__process = multiprocessing.Process()
        self.__queue = multiprocessing.Queue()

    def __call__(self, *args, **kwargs):
        """Execute the embedded function object asynchronously.

        The function given to the constructor is transparently called and
        requires that "ready" be intermittently polled. If and when it is
        True, the "value" property may then be checked for returned data.
        """
        self.__limit = kwargs.pop('timeout', self.__limit)
        self.__queue = multiprocessing.Queue(1)
        args = (self.__queue, self.__function) + args
        self.__process = multiprocessing.Process(target=_target,
                                                 args=args,
                                                 kwargs=kwargs)
        self.__process.daemon = True
        self.__process.start()
        self.__timeout = self.__limit + time.time()
        while not self.ready:
            time.sleep(0.01)
        return self.value

    def cancel(self):
        """Terminate any possible execution of the embedded function."""
        if self.__process.is_alive():
            self.__process.terminate()

        _raise_exception(self.__timeout_exception, self.__exception_message)

    @property
    def ready(self):
        """Read-only property indicating status of "value" property."""
        if self.__timeout < time.time():
            self.cancel()
        return self.__queue.full() and not self.__queue.empty()

    @property
    def value(self):
        """Read-only property containing data returned from function."""
        if self.ready is True:
            flag, load = self.__queue.get()
            if flag:
                return load
            raise load
