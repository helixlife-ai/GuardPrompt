# GuardPrompt
A Python library to protect prompt engineering against theft

## Usage
```py
from guard_prompt import LLMResponseWrapper

res = mock_get_llm_response()

llm_response = LLMResponseWrapper(
    data=response.iter_content  # Generator
)
# or
llm_response = LLMResponseWrapper(
    data=response.aiter_content  # AsyncGenerator
)
# or
llm_response = LLMResponseWrapper(
    data=res.json()["choices"][0]["message"]["content"]  # text
)

for chunk in llm_response.iter_response():
    ...

```
