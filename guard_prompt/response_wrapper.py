from collections.abc import AsyncGenerator, Generator
from typing import Union

from ._exceptions import InvalidResponseError, NoneResponseError
from ._utils import generator_reader, get_lcs


class ResponseWrapper:
    def __init__(
        self,
        system_prompt: str,
        data: Union[
            str,
            Generator[str, None, None],
            AsyncGenerator[str, None],
            None,
        ] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.data = data

    def set_response(
        self,
        response: Union[
            str,
            Generator[str, None, None],
            AsyncGenerator[str, None],
        ],
    ):
        self.data = response

    # TEST: iter_response()
    def iter_response(
        self,
        chunk_size: int = 500,
    ) -> Generator[str, None, None]:
        """
        Args:
            chunk_size (int, optional):
                The length of string chunk.
                The larger the chunk_size, the better the inspection result,
                but streaming response may become slower.
                Defaults to 500.

        Raises:
            NoneResponseError: Raise when the response.data not set yet.
            InvalidResponseError: Raise when response.data's type
                not belongs to [`str`, `Generator`, `AsyncGenerator`]

        Yields:
            str: The response chunk

        """
        if self.data is None:
            raise NoneResponseError()

        string_buffer = []
        for item in generator_reader(self.data, chunk_size=chunk_size):
            yield item
            string_buffer.append(item)
            if len(string_buffer) > chunk_size and (
                length := get_lcs(self.system_prompt, string_buffer)
            ):
                # OPTIMIZE: 如何判断 string_buffer 是非法的
                if length > len(string_buffer) * 0.8:
                    raise InvalidResponseError()
                string_buffer.clear()
