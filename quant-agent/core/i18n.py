from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

OutputLanguage: TypeAlias = Literal["en", "zh", "bilingual"]

DEFAULT_OUTPUT_LANGUAGE: OutputLanguage = "bilingual"
SUPPORTED_OUTPUT_LANGUAGES = ("en", "zh", "bilingual")

_LANGUAGE_ALIASES: dict[str, OutputLanguage] = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "zh-hans": "zh",
    "zh_hans": "zh",
    "bilingual": "bilingual",
    "both": "bilingual",
    "dual": "bilingual",
    "zh-en": "bilingual",
    "zh_en": "bilingual",
    "cn-en": "bilingual",
    "cn_en": "bilingual",
}


@dataclass(frozen=True, slots=True)
class LocalizedText:
    """Chinese and English wording for human-facing project output."""

    en: str
    zh: str

    def render(
        self,
        language: str | None = None,
        *,
        separator: str = " / ",
    ) -> str:
        return render_text(self.en, self.zh, language, separator=separator)


def normalize_output_language(
    language: str | None,
    *,
    default: OutputLanguage = DEFAULT_OUTPUT_LANGUAGE,
) -> OutputLanguage:
    """Normalize user-facing output language names and aliases."""

    if language is None:
        return default
    normalized = language.strip().lower()
    if not normalized:
        return default
    if normalized not in _LANGUAGE_ALIASES:
        supported = ", ".join(SUPPORTED_OUTPUT_LANGUAGES)
        msg = f"Unsupported output_language: {language}. Use one of: {supported}."
        raise ValueError(msg)
    return cast(OutputLanguage, _LANGUAGE_ALIASES[normalized])


def render_text(
    en: str,
    zh: str,
    language: str | None = None,
    *,
    separator: str = " / ",
) -> str:
    """Render one human-facing text value in English, Chinese, or both."""

    output_language = normalize_output_language(language)
    if output_language == "en":
        return en
    if output_language == "zh":
        return zh
    return f"{zh}{separator}{en}"


def render_label(en: str, zh: str, language: str | None = None) -> str:
    """Render a compact UI or Markdown label."""

    return render_text(en, zh, language, separator=" / ")


def render_paragraph(en: str, zh: str, language: str | None = None) -> str:
    """Render a longer paragraph without cramming both languages on one line."""

    return render_text(en, zh, language, separator="\n\n")
