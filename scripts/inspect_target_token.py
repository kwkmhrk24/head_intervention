import argparse

from transformers import AutoTokenizer


MISTRAL_TARGET_TOKEN_ID = 12875
TAIL_TOKEN_COUNT = 16
TAIL_CHARACTER_COUNT = 160


def get_args():
    parser = argparse.ArgumentParser(
        description="Inspect Mistral target-token tokenization without loading the model."
    )
    parser.add_argument("--model", required=True, help="Model/tokenizer path.")
    parser.add_argument(
        "--strings",
        nargs="+",
        required=True,
        help=r"Strings to inspect. CLI escapes such as \n, \r, and \t are expanded.",
    )
    return parser.parse_args()


def expand_cli_escapes(value):
    return value.replace(r"\n", "\n").replace(r"\r", "\r").replace(r"\t", "\t")


def encode(tokenizer, text, add_special_tokens):
    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


def decode_token(tokenizer, token_id):
    return tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def print_tokenization(tokenizer, text, add_special_tokens, indent=""):
    token_ids = encode(tokenizer, text, add_special_tokens=add_special_tokens)
    print(f"{indent}add_special_tokens={add_special_tokens}")
    print(f"{indent}token_count={len(token_ids)}")
    print(f"{indent}token_ids={token_ids}")
    print(f"{indent}contains_{MISTRAL_TARGET_TOKEN_ID}={MISTRAL_TARGET_TOKEN_ID in token_ids}")
    print(f"{indent}individual_decodes:")
    for token_id in token_ids:
        print(f"{indent}  {token_id}: {decode_token(tokenizer, token_id)!r}")
    return token_ids


def print_chat_template_diagnostics(tokenizer):
    print("\n=== chat template ===")
    messages = [{"role": "user", "content": "Hello"}]
    try:
        templated = tokenizer.apply_chat_template(messages, tokenize=False)
    except (AttributeError, ValueError, TypeError) as exc:
        print(f"unavailable: {type(exc).__name__}: {exc}")
        return

    for label, text in (
        ("templated", templated),
        ("templated_plus_space", templated + " "),
    ):
        token_ids = encode(tokenizer, text, add_special_tokens=False)
        tail_ids = token_ids[-TAIL_TOKEN_COUNT:]
        print(f"{label}_tail_text={text[-TAIL_CHARACTER_COUNT:]!r}")
        print(f"{label}_token_count={len(token_ids)}")
        print(f"{label}_tail_token_ids={tail_ids}")
        print(
            f"{label}_tail_individual_decodes="
            f"{[(token_id, decode_token(tokenizer, token_id)) for token_id in tail_ids]}"
        )
        print(f"{label}_contains_{MISTRAL_TARGET_TOKEN_ID}={MISTRAL_TARGET_TOKEN_ID in token_ids}")


def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"tokenizer={tokenizer.__class__.__name__}")
    print(f"target_token_id={MISTRAL_TARGET_TOKEN_ID}")
    print(f"target_token_decode={decode_token(tokenizer, MISTRAL_TARGET_TOKEN_ID)!r}")

    for raw_text in args.strings:
        text = expand_cli_escapes(raw_text)
        print(f"\n=== input {text!r} ===")
        print_tokenization(tokenizer, text, add_special_tokens=False)
        print_tokenization(tokenizer, text, add_special_tokens=True)

    print_chat_template_diagnostics(tokenizer)


if __name__ == "__main__":
    main()
