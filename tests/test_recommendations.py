from style_and_sense.recommendations import (
    CHAT_COMPLETIONS_MODE,
    DEFAULT_DARTMOUTH_MODEL,
    RESPONSES_MODE,
    generate_fallback_outfits,
    get_llm_config,
    is_complete_outfit,
    parse_json_object,
    parse_temperature,
    validate_minimum_closet,
    validate_outfits,
)


def garment(garment_id, category, subcategory):
    return {
        "id": garment_id,
        "category": category,
        "subcategory": subcategory,
        "colors": ["black"],
        "style_tags": ["casual"],
        "season_tags": [],
        "formality": "casual",
        "caption": subcategory,
        "laundry_status": "available",
    }


def test_validate_outfits_rejects_hallucinated_ids():
    candidates = [
        garment("top_1", "top", "t-shirt"),
        garment("bottom_1", "bottom", "jeans"),
    ]
    raw = [
        {
            "title": "Bad Outfit",
            "items": [
                {"id": "top_1", "role": "top"},
                {"id": "missing", "role": "bottom"},
            ],
            "explanation": "Nope.",
            "confidence": 0.9,
        }
    ]

    assert validate_outfits(raw, candidates) == []


def test_validate_outfits_accepts_complete_outfit_without_shoes_when_no_shoes_available():
    candidates = [
        garment("top_1", "top", "t-shirt"),
        garment("bottom_1", "bottom", "jeans"),
    ]
    raw = [
        {
            "title": "Simple Outfit",
            "items": [
                {"id": "top_1", "role": "top"},
                {"id": "bottom_1", "role": "bottom"},
            ],
            "explanation": "A wearable simple outfit.",
            "confidence": 0.9,
        }
    ]

    outfits = validate_outfits(raw, candidates)

    assert len(outfits) == 1
    assert outfits[0].item_ids == ["top_1", "bottom_1"]


def test_complete_outfit_requires_shoes_when_shoes_available():
    outfit = [
        garment("top_1", "top", "t-shirt"),
        garment("bottom_1", "bottom", "jeans"),
    ]

    assert not is_complete_outfit(
        outfit,
        available_categories={"top", "bottom", "shoes"},
    )


def test_fallback_outfits_builds_complete_options():
    candidates = [
        garment("top_1", "top", "t-shirt"),
        garment("bottom_1", "bottom", "jeans"),
        garment("shoes_1", "shoes", "sneakers"),
    ]

    raw = generate_fallback_outfits(
        user_prompt="class",
        candidate_garments=candidates,
        style_rules=[],
    )

    assert raw
    assert {item["id"] for item in raw[0]["items"]} == {
        "top_1",
        "bottom_1",
        "shoes_1",
    }


def test_validate_minimum_closet_reports_missing_base():
    assert validate_minimum_closet([garment("shoes_1", "shoes", "sneakers")])


def test_parse_temperature():
    assert parse_temperature("it is 65 degrees") == 65
    assert parse_temperature("rainy") is None


def test_parse_json_object_accepts_markdown_fence():
    parsed = parse_json_object('```json\n{"outfits":[]}\n```')

    assert parsed == {"outfits": []}


def test_get_llm_config_defaults_to_responses_for_direct_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_MODE", raising=False)

    config = get_llm_config()

    assert config.api_mode == RESPONSES_MODE
    assert config.model == "gpt-4o-mini"


def test_get_llm_config_defaults_to_chat_completions_for_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://chat.dartmouth.edu/api")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_MODE", raising=False)

    config = get_llm_config()

    assert config.api_mode == CHAT_COMPLETIONS_MODE
    assert config.model == DEFAULT_DARTMOUTH_MODEL


def test_get_llm_config_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_MODE", "unknown")

    try:
        get_llm_config()
    except ValueError as exc:
        assert "Unsupported OPENAI_API_MODE" in str(exc)
    else:
        raise AssertionError("Expected invalid API mode to raise ValueError.")
