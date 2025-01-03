class InvalidResponseError(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(f"Invalid response: {message if message else ''}")


class NoneResponseError(Exception):
    def __init__(self):
        super().__init__(
            "LLMResponse.data is None, "
            "maybe forgot to assign a value to `response.data`?"
        )


class NotSupportedResponseTypeError(Exception):
    def __init__(self, type_name: str):
        super().__init__(
            f"LLMResponse.data should be `str` or `Generator` or `AsyncGenerator`, "
            f"got {type_name}."
        )
