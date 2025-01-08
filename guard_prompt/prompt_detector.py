import os
from typing import Literal

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# model_id = "meta-llama/Prompt-Guard-86M"
DEFAULT_MODEL_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/model"


class PromptDetector:
    def __init__(
        self,
        model_id_or_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id_or_path
        )

    def _get_score(
        self,
        text: str,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> tuple[float, float]:
        """
        Evaluate the model on the given text with temperature-adjusted softmax.
        Note, as this is a DeBERTa model, the input text should have a maximum length of 512.

        Args:
            text (str): The input text to classify.
            temperature (float): The temperature for the softmax function. Default is 1.0.
            device (str): The device to evaluate the model on.

        Returns:
            # torch.Tensor: The probability of each class adjusted by the temperature.
            tuple[float, float]: indirect_injection score, jailbreak score
        """
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
            print("injection_score:", injection_score)
            print("jailbreak_score:", jailbreak_score)
            print()
            if jailbreak_score > threshold:
                return False, "jailbreak"
            elif from_ == "indirect" and injection_score > threshold:
                return False, "injection"

        return True, ""
