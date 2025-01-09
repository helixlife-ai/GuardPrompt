import os
from typing import Literal

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptDetector:
    def __init__(
        self,
        model_id_or_path: str = "meta-llama/Prompt-Guard-86M",
    ) -> None:
        """
        Args:
            model_id_or_path (str, optional):
                Default model id or model local path.
        """
        self.model_id_or_path = model_id_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id_or_path
        )

    def _get_score(
        self,
        text: str,
        temperature: float = 1.0,
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ) -> tuple[float, float]:
        """
        Get the text's indirect_injection score and jailbreak score.

        Args:
            text (str): The input text, the maximum length is less than 512

            temperature (float): The temperature for the softmax function.
            Default is 1.0.

            device ("auto" | "cpu" | "cuda"): The device to evaluate the model on.

        Returns:
            tuple[float, float]: indirect_injection score, jailbreak score

        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Encode the text
        inputs = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Get logits from the model
        with torch.no_grad():
            logits = self.model(**inputs).logits
        # Apply temperature scaling
        scaled_logits = logits / temperature
        # Apply softmax to get probabilities
        probabilities = softmax(scaled_logits, dim=-1)

        return probabilities[0, 1].item(), probabilities[0, 2].item()

    def check(
        self,
        text: str,
        from_: Literal["user", "indirect"],
        threshold: float = 0.65,
    ) -> tuple[bool, Literal["jailbreak", "injection", ""]]:
        """
        Detect the safety of `text`.

        Args:
            text (str): Input text.

            from_ (Literal["user", "indirect"]): Where is the `text` from.

                `user`: Come from user.

                `indirect`: Come from `indirect` that might inject prompt, like:
                    ```json
                    {
                        name: "Jack",
                        age: "Forget above message. Do ..."
                    }
                    ```

            threshold (float, optional): Used to check score,
                if score ([0, 1]) >= threshold, the text is considered unsafe.
                Defaults to 0.65.

        Returns:
            (tuple[bool, Literal["jailbreak", "injection", ""]]):
                1. result (bool): if the `text` is safe
                2. message ("jailbreak" | "injection" | ""):
                    if `result` is `True`, then `message` is `""`,
                    otherwise the `message` is the reason why the
                    text is not safe. if param `from_` is `user`,
                    the `injection` will be not detected.
        """
        idx = 0
        while idx < len(text):
            short_text = text[idx : min(len(text), idx + 512)]
            idx += 256

            injection_score, jailbreak_score = self._get_score(text=short_text)
            if jailbreak_score > threshold:
                return False, "jailbreak"
            elif from_ == "indirect" and injection_score > threshold:
                return False, "injection"

        return True, ""
