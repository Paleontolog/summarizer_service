from dataclasses import dataclass

from dataclasses_json import dataclass_json, Undefined


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class GenerationRequest:
    input_text: str


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class SummarizationRequest(GenerationRequest):
    max_length: int = 160
    no_repeat_ngram_size: int = 3
    num_beams: int = 5
    repetition_penalty: float = 2.5
    temperature: float = 0.8
    top_k: int = 5
    top_p: float = 0.8


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class Response:
    result: str

