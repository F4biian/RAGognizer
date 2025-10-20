from abc import ABC, abstractmethod


class HallucinationDetector(ABC):
    def __init__(self, name: str, aggregate_response_level_from_token_level: bool=False):
        self.name = name
        self.aggregate_response_level_from_token_level = aggregate_response_level_from_token_level

    def _ensure_tokens_in_response(
        self,
        response: str,
        token_prediction: list[dict[str, int | str | float]],
        token_key: str = "text"
    ) ->  list[dict[str, int | str | float]]:
        # Remove tokens that are not part of the real response (e.g. "[SEP]" or additional whitespace)
        final_pred = []
        last_end = 0
        left_over = response
        for tok in token_prediction:
            tok_str = tok[token_key]

            if len(tok_str) > 0 and len(left_over) > 0:
                try:
                    starts_at = left_over.index(tok_str)
                    ends_at = starts_at + len(tok_str)

                    tok_str = left_over[:ends_at]
                    tok[token_key] = tok_str
                    tok["start"]   = last_end
                    tok["end"]     = last_end + len(tok_str)

                    left_over = left_over[ends_at:]
                    last_end = tok["end"]

                    final_pred.append(tok)
                except:
                    raise Exception(f"Could not find '{tok_str}' in '{left_over}'")

        return final_pred

    def _aggregate_to_response_level(
        self,
        token_level_pred: list[dict[str, float | str | int]] | None,
    ) -> dict[str, float | int] | None:
        if token_level_pred is None:
            return None
        
        max_token = None
        for tok in token_level_pred:
            if max_token is None or max_token["prob"] < tok["prob"]:
                max_token = tok

        if max_token is None:
            raise ValueError(f"Could not find the maximum token in the following token-level prediction: {token_level_pred}")

        return {
            "prob": max_token["prob"],
            "pred": max_token["pred"],
        }

    def predict_dataset(
        self,
        data: list
    ) -> list[list[dict[str, float | str | int]] | dict[str, float | int] | None]:
        results = []
        for entry in data:
            results.append(self.predict(**entry))
        return results

    @abstractmethod
    def predict(
        self,
        documents: list[str] = None,
        document: str = None,
        user: str = None,
        response: str = None,
        chat: list[dict[str, str]] = None,
        token_level: bool=True,
        **kwargs,
    ) -> list[dict[str, float | str | int]] | dict[str, float | int] | None:
        pass