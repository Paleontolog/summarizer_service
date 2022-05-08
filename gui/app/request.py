import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class GenerationRequest:
    input_text: str


@dataclass_json
@dataclass
class SummarizationRequest(GenerationRequest):
    max_length: int = 160
    no_repeat_ngram_size: int = 3
    num_beams: int = 5
    repetition_penalty: float = 2.5
    temperature: float = 0.8
    top_k: int = 5
    top_p: float = 0.8


