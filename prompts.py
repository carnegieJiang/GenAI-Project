from typing import Optional


def build_prompt(style_label: Optional[str], instruction: Optional[str]) -> str:
    if instruction:
        cleaned = instruction.strip()
        if cleaned:
            return cleaned

    if style_label:
        return f"restyle this image as {style_label.strip()} while preserving the original content"

    return "restyle this image while preserving the original content"
