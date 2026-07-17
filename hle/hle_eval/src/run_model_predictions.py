import os
import json
import argparse
import asyncio
import re
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from subset_utils import filter_questions_by_subset, load_subset_file
from judge_prompts import COMPATIBLE_JUDGE_PROMPT, build_judge_response, parse_compatible_judge_json

client = None
judge_client = None

SYSTEM_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def normalize_answer(value):
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("**", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r`'\".,;:!?()[]{}")
    return text


def extract_final_answer(response_text):
    if not response_text:
        return ""
    # Prefer the explicit "Answer:" field in the prompted format.
    match = re.search(r"(?im)^answer\s*:\s*(.+)$", response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def is_em_correct(question, response_text):
    gold = normalize_answer(question.get("answer"))
    pred = normalize_answer(extract_final_answer(response_text))
    if not gold or not pred:
        return None
    return pred == gold

def create_client(args):
    api_key = args.api_key
    if api_key is None and args.api_key_env:
        api_key = os.environ.get(args.api_key_env)
    if api_key is None and args.base_url:
        api_key = "EMPTY"

    return AsyncOpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


def create_judge_client(args):
    api_key = args.judge_api_key
    if api_key is None and args.judge_api_key_env:
        api_key = os.environ.get(args.judge_api_key_env)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=args.judge_base_url,
        timeout=args.judge_timeout,
        max_retries=args.judge_max_retries,
    )

def format_message(question):
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question.get('image'): # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(question):
    messages = format_message(question)
    try:
        request_kwargs = {
            "model": args.model,
            "messages": messages,
            "stream": False,
        }
        if args.max_completion_tokens is not None:
            request_kwargs[args.token_param] = args.max_completion_tokens
        if "o1" not in args.model:
            request_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                request_kwargs["top_p"] = args.top_p
            if args.presence_penalty is not None:
                request_kwargs["presence_penalty"] = args.presence_penalty

        extra_body = {}
        if args.top_k is not None:
            extra_body["top_k"] = args.top_k
        if args.min_p is not None:
            extra_body["min_p"] = args.min_p
        if args.repetition_penalty is not None:
            extra_body["repetition_penalty"] = args.repetition_penalty
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        create_coro = client.chat.completions.create(**request_kwargs)
        q_timeout = getattr(args, "question_timeout", None)
        if q_timeout is not None and q_timeout > 0:
            response = await asyncio.wait_for(create_coro, timeout=float(q_timeout))
        else:
            response = await create_coro
        message = response.choices[0].message
        content = message.content
        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None and getattr(message, "model_extra", None):
            reasoning_content = message.model_extra.get("reasoning_content")
        tokens = json.loads(response.usage.model_dump_json()) if response.usage else {}
    except asyncio.TimeoutError:
        print(
            f"Error: question wall-clock timeout ({getattr(args, 'question_timeout', None)}s) "
            f"id={question.get('id')}",
            flush=True,
        )
        return None
    except Exception as e:
        print("Error:", e, flush=True)
        return None
    
    if content is None: # failed
        return None
    
    em_correct = is_em_correct(question, content)

    judge_response = None
    judge_correct = None
    if args.realtime_judge:
        prompt = COMPATIBLE_JUDGE_PROMPT.format(
            question=question["question"],
            response=content,
            correct_answer=question["answer"],
        )
        try:
            judge_completion = await judge_client.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                stream=False,
            )
            raw = (judge_completion.choices[0].message.content or "").strip()
            parsed = parse_compatible_judge_json(raw)
            judge_response, judge_correct = build_judge_response(
                question["answer"],
                parsed,
                judge_model=args.judge_model,
            )
        except Exception as e:
            judge_response = {
                "correct_answer": question["answer"],
                "model_answer": None,
                "reasoning": f"realtime_judge_error: {e}",
                "correct": "no",
                "confidence": 100,
                "judge_model": args.judge_model,
            }
            judge_correct = None

    return {
        "id": question["id"],
        "response": content,
        "reasoning_content": reasoning_content,
        "usage": tokens,
        "em_correct": em_correct,
        "judge_response": judge_response,
        "judge_correct": judge_correct,
    }


async def attempt_all(questions, predictions, output_filepath):
    async def bound_func(question):
        async with semaphore:
            return await attempt_question(question)
            
    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [asyncio.create_task(bound_func(q)) for q in questions]
    total = len(tasks)
    completed = 0
    failures = 0
    scored = 0
    scored_correct = 0
    bar = tqdm(
        total=total,
        position=args.progress_position,
        leave=True,
        dynamic_ncols=True,
        desc=f"[{args.model}] acc=0.00% (0/0)",
    )

    for task in asyncio.as_completed(tasks):
        result = await task
        completed += 1
        if result is None:
            failures += 1
        else:
            unique_id = result["id"]
            predictions[unique_id] = {
                "model": args.model,
                "response": result["response"],
                "usage": result["usage"],
            }
            if result["reasoning_content"] is not None:
                predictions[unique_id]["reasoning_content"] = result["reasoning_content"]
            if args.realtime_judge and result["judge_response"] is not None:
                predictions[unique_id]["judge_response"] = result["judge_response"]
            save_predictions(output_filepath, predictions)

            metric_correct = result["judge_correct"] if args.realtime_judge else result["em_correct"]
            if metric_correct is not None:
                scored += 1
                if metric_correct:
                    scored_correct += 1

        live_acc = (100.0 * scored_correct / scored) if scored else 0.0
        metric_name = "judge_acc" if args.realtime_judge else "em_acc"
        bar.set_description(
            f"[{args.model}] {metric_name}={live_acc:.2f}% ({scored_correct}/{scored}) fail={failures}"
        )
        bar.update(1)

    bar.close()
    final_acc = (100.0 * scored_correct / scored) if scored else 0.0
    metric_name = "final_live_judge_acc" if args.realtime_judge else "final_live_em_acc"
    print(
        f"[{args.model}] done | total={total} ok={total - failures} fail={failures} "
        f"| {metric_name}={final_acc:.2f}% ({scored_correct}/{scored})",
        flush=True,
    )


def save_predictions(output_filepath, predictions):
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
        f.flush()
        os.fsync(f.fileno())

    
def main(args):
    global client, judge_client
    assert args.num_workers >= 1, "num_workers must be 1 or greater"
    client = create_client(args)
    if args.realtime_judge:
        judge_client = create_judge_client(args)

    dataset = load_dataset(args.dataset, split="test").to_dict() 

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    if args.text_only:
        questions = [q for q in questions if not q.get("image")]

    if args.subset_file:
        subset_ids = load_subset_file(args.subset_file)
        questions = filter_questions_by_subset(questions, subset_ids)
    elif args.max_samples:
        questions = questions[:args.max_samples]
    
    output_filepath = args.output or f"hle_{os.path.basename(args.model)}.json"


    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    print(
        f"[{args.model}] start | pending={len(questions)} "
        f"| num_workers={args.num_workers} | output={output_filepath} "
        f"| realtime_judge={args.realtime_judge} "
        f"| question_timeout={getattr(args, 'question_timeout', None)}",
        flush=True,
    )

    asyncio.run(attempt_all(questions, predictions, output_filepath))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="HLE HF Dataset")
    parser.add_argument("--model", type=str, help="Model Endpoint Name")
    parser.add_argument("--max_completion_tokens", type=int, default=None, help="Limit completion tokens. Recommended to avoid model collapse.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling (SGLang-compatible via extra_body).")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling floor (SGLang-compatible via extra_body).")
    parser.add_argument("--presence_penalty", type=float, default=None, help="Presence penalty.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty (SGLang-compatible via extra_body).")
    parser.add_argument("--num_workers", type=int, default=10, help="Async semaphore size. This depends on your rate limit.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation to first N samples. Ignored when --subset_file is set.")
    parser.add_argument("--subset_file", type=str, default=None, help="JSON file containing a fixed list of question ids to evaluate.")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible base URL for the model endpoint.")
    parser.add_argument("--api_key", type=str, default=None, help="API key for the model endpoint. Prefer --api_key_env to avoid shell history.")
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY", help="Environment variable containing the model endpoint API key.")
    parser.add_argument("--output", type=str, default=None, help="Output path for model predictions.")
    parser.add_argument("--text_only", action="store_true", help="Skip questions with images.")
    parser.add_argument("--token_param", choices=["max_completion_tokens", "max_tokens"], default="max_completion_tokens", help="Token limit parameter accepted by the endpoint.")
    parser.add_argument("--timeout", type=float, default=600.0, help="OpenAI/httpx idle timeout in seconds.")
    parser.add_argument(
        "--question_timeout",
        type=float,
        default=1800.0,
        help="Hard wall-clock timeout per question in seconds (cancels hung long generations).",
    )
    parser.add_argument("--max_retries", type=int, default=1, help="OpenAI client max retries.")
    parser.add_argument("--progress_position", type=int, default=0, help="Terminal row offset for tqdm bar.")
    parser.add_argument("--realtime_judge", action="store_true", help="Judge each response immediately and persist judge result.")
    parser.add_argument("--judge_model", type=str, default="deepseek-v4-flash", help="Judge model for realtime scoring.")
    parser.add_argument("--judge_base_url", type=str, default="https://api.deepseek.com", help="Judge OpenAI-compatible base URL.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Judge API key. Prefer --judge_api_key_env.")
    parser.add_argument("--judge_api_key_env", type=str, default="DEEPSEEK_API_KEY", help="Env var containing judge API key.")
    parser.add_argument("--judge_timeout", type=float, default=180.0, help="Judge client timeout in seconds.")
    parser.add_argument("--judge_max_retries", type=int, default=1, help="Judge client max retries.")
    args = parser.parse_args()
    main(args)
