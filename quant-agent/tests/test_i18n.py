from __future__ import annotations

import pytest

from core.i18n import LocalizedText, normalize_output_language, render_label


def test_normalize_output_language_accepts_aliases() -> None:
    assert normalize_output_language("en") == "en"
    assert normalize_output_language("zh-CN") == "zh"
    assert normalize_output_language("both") == "bilingual"


def test_render_label_supports_english_chinese_and_bilingual() -> None:
    assert render_label("Report", "报告", "en") == "Report"
    assert render_label("Report", "报告", "zh") == "报告"
    assert render_label("Report", "报告", "bilingual") == "报告 / Report"


def test_localized_text_rejects_unknown_language() -> None:
    text = LocalizedText(en="Report", zh="报告")

    with pytest.raises(ValueError, match="Unsupported output_language"):
        text.render("fr")
