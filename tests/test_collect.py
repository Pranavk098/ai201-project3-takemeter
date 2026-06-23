from scripts.collect import clean_text, is_filterable


def test_clean_text_collapses_whitespace():
    assert clean_text("  hello   world\n\n") == "hello world"


def test_is_filterable_drops_deleted_short_and_urls():
    assert is_filterable("[deleted]") is True
    assert is_filterable("[removed]") is True
    assert is_filterable("nice") is True            # too short (<15)
    assert is_filterable("https://i.imgur.com/x") is True
    assert is_filterable("") is True


def test_is_filterable_keeps_real_comment():
    assert is_filterable("Leclerc lost that on strategy, not pace.") is False
