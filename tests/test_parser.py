from hierarchical_soft_prompt.parser import parse_json_output


def test_parser_accepts_fenced_json():
    parsed, ok = parse_json_output('```json\n{"value": 3}\n```')
    assert ok
    assert parsed == {"value": 3}


def test_parser_rejects_non_json():
    parsed, ok = parse_json_output("not json")
    assert not ok
    assert parsed == {}
