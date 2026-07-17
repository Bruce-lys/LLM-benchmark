import argparse
import os
from openai import OpenAI


def get_reasoning_content(message):
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content is None and getattr(message, "model_extra", None):
        reasoning_content = message.model_extra.get("reasoning_content")
    return reasoning_content


def main(args):
    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    if api_key is None:
        api_key = "EMPTY"

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        temperature=0,
        stream=False,
    )

    message = response.choices[0].message
    reasoning_content = get_reasoning_content(message)
    print("content:")
    print(message.content)
    print("\nreasoning_content_present:", bool(reasoning_content))
    if reasoning_content:
        print("\nreasoning_content_preview:")
        print(reasoning_content[: args.preview_chars])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", type=str, required=True, help="SGLang OpenAI-compatible base URL.")
    parser.add_argument("--model", type=str, required=True, help="Served model name.")
    parser.add_argument("--api_key", type=str, default=None, help="API key. SGLang usually accepts any non-empty value.")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--prompt", type=str, default="请计算 19 * 23，并给出最终答案。", help="Prompt used to check thinking output.")
    parser.add_argument("--timeout", type=float, default=600.0, help="OpenAI client timeout in seconds.")
    parser.add_argument("--max_retries", type=int, default=1, help="OpenAI client max retries.")
    parser.add_argument("--preview_chars", type=int, default=1000, help="Number of reasoning characters to print.")
    args = parser.parse_args()
    main(args)
