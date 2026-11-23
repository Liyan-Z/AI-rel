"""Prompt construction. The instruction stays in one language for all variants so only the question varies."""
from __future__ import annotations

INSTR = {
    "en": "Answer the following multiple-choice question. Reply with only the letter of the correct option.",
    "ar": "أجب عن سؤال الاختيار من متعدد التالي. اكتب حرف الإجابة الصحيحة فقط.",
}


def build_prompt(question: str, choices: list[str], labels: list[str], lang: str = "en") -> str:
    lines = [INSTR[lang], "", question.strip(), ""]
    lines += [f"{lab}. {str(c).strip()}" for lab, c in zip(labels, choices)]
    lines += ["", "Answer:" if lang == "en" else "الإجابة:"]
    return "\n".join(lines)
