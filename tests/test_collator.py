from hierarchical_soft_prompt.collator import SoftPromptCollator


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True):
        prefix = [2] if add_special_tokens else []
        return {"input_ids": prefix + [3 + (ord(char) % 50) for char in text]}


def test_collator_masks_prompt_padding_and_soft_tokens():
    collator = SoftPromptCollator(
        tokenizer=CharacterTokenizer(),
        n_soft_tokens=3,
        max_length=64,
    )
    batch = collator(
        [
            {
                "input_text": "procedure:",
                "target_json_str": '{"value": 1}',
                "target_json": {"value": 1},
            },
            {
                "input_text": "short:",
                "target_json_str": '{"value": 2}',
                "target_json": {"value": 2},
            },
        ]
    )

    assert batch["attention_mask"].shape[1] == batch["input_ids"].shape[1] + 3
    assert (batch["labels"][:, :3] == -100).all()
    assert (batch["labels"][0, 3 : 3 + batch["prompt_lengths"][0]] == -100).all()


def test_evaluation_prompt_does_not_depend_on_gold_length():
    collator = SoftPromptCollator(
        tokenizer=CharacterTokenizer(),
        include_targets=False,
        max_prompt_length=8,
    )
    common = {"input_text": "same procedure", "target_json": {}}
    batch = collator(
        [
            {**common, "target_json_str": "{}"},
            {**common, "target_json_str": '{"much": "longer target"}'},
        ]
    )

    assert batch["prompt_lengths"].tolist() == [8, 8]
    assert (batch["input_ids"][0] == batch["input_ids"][1]).all()
