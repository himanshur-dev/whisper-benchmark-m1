from dataclasses import dataclass

import jiwer
from whisper.normalizers import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


@dataclass
class EvaluationResult:
    wer: float
    cer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    ref_word_count: int
    rtf: float


def evaluate(
    hypothesis: str,
    reference: str,
    inference_time_s: float,
    audio_duration_s: float,
) -> EvaluationResult:
    ref_norm = _normalizer(reference)
    hyp_norm = _normalizer(hypothesis)

    measures = jiwer.process_words(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)
    rtf = inference_time_s / audio_duration_s if audio_duration_s > 0 else 0.0

    return EvaluationResult(
        wer=measures.wer,
        cer=cer,
        substitutions=measures.substitutions,
        deletions=measures.deletions,
        insertions=measures.insertions,
        hits=measures.hits,
        ref_word_count=measures.hits + measures.substitutions + measures.deletions,
        rtf=rtf,
    )
