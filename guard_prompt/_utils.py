import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import TypeVar
from typing import Union
from collections.abc import Sequence
from collections.abc import Coroutine
from collections.abc import Generator
from collections.abc import AsyncGenerator

from ._exceptions import NotSupportedResponseTypeError


T = TypeVar("T")


#  TODO: 传参字符串数组
def get_lcs(str1: str | Sequence[str], str2: str | Sequence[str]) -> int:
    """
    get length of the longest common substring.

    Time complexity: O(len(str1) * len(str2))

    Args:
        str1 (str | Sequence[str]):

        str2 (str | Sequence[str]):

    Returns:
        int: the length of the longest common substring

    """
    def format_param(param: str | Sequence[str]) -> list[str]:
        if isinstance(param, str):
            return list(param)
        elif isinstance(param, Sequence):
            res = []
            for item in param:
                if not isinstance(item, str):
                    raise TypeError("sequence param must be Sequence[str]")
                res.extend(list(item))
            return res
        else:
            raise TypeError("param must be str or Sequence[str]")

    pattern = format_param(str1)
    matching = format_param(str2)
    len1, len2 = len(pattern), len(matching)

    # dp[i][j]: The length of the LCS of `pattern[:i-1]`` and `matching[:j-1]`
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_length = 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if pattern[i - 1] == matching[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1

                if dp[i][j] > max_length:
                    max_length = dp[i][j]
            else:
                dp[i][j] = 0

    return max_length



def run_coroutine_sync(coroutine: Coroutine[Any, Any, T], timeout: float = 30) -> T:
    """
    run coroutine on sync function.

    by https://stackoverflow.com/a/78911765

    """
    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()


def generator_reader(
    data: Union[
        str,
        Generator[str, None, None],
        AsyncGenerator[str, None],
    ],
    chunk_size: int,
):
    """
    convert [str | Generator | AsyncGenerator] to [Generator].

    Args:
        data (str | Generator | AsyncGenerator):
        chunk_size (int): the max chunk size.

    Raises:
        TypeError: raise when data's type not belongs to [str, Generator, AsyncGenerator].
        StopIteration: raise when the generator is done.

    Yields:
        Any: chunk

    """
    if isinstance(data, Generator):
        yield from data

    elif isinstance(data, AsyncGenerator):
        try:
            while True:
                yield run_coroutine_sync(anext(data))
        except StopAsyncIteration:
            raise StopIteration()

    elif isinstance(data, str):
        for i in range(len(data)):
            if i + chunk_size >= len(data):
                yield data[i:]
                break
            yield data[i:i + chunk_size]
        raise StopIteration()

    else:
        raise NotSupportedResponseTypeError(type(data))
