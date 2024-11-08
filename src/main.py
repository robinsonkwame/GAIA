import json
import pandas as pd
from dotenv import load_dotenv
import argparse
import asyncio

from .utils import load_dataset_by_type
from .agent import initialize
from .evaluator import initialize_evaluator, evaluate_response

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, required=False, help="Type of task to test (e.g. GSM8K, HotpotQA-easy)", default="GSM8K")
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples to run")
    parser.add_argument(
        "--example_offsets",
        type=lambda s: [int(x) for x in s.split(',')],  # Convert comma-separated string to list of ints
        help="Comma-separated list of example indices to run (e.g., '0,1,2')",
        default=None
    )
    args = parser.parse_args()
    return args

async def run_single_example(example, task_type, question_offset):
    """Run a single example with a fresh agent instance and evaluate the response"""
    gaia_agent = initialize()
    evaluator_client = initialize_evaluator()
    
    # Configure system prompt based on task type
    if task_type == "GSM8K":
        system_prompt = "You are a mathematical reasoning expert."
    elif task_type in ["HotpotQA-easy", "HotpotQA-medium"]:
        system_prompt = "You are an internet research expert."
    else:
        system_prompt = "You are an expert at complex reasoning and internet research."
        
    gaia_agent.global_context = system_prompt
    
    # Run the example and get full response including intermediate steps
    response = gaia_agent.run(example["question"])
    
    # Evaluate the response
    evaluation = await evaluate_response(evaluator_client, response, example)
    
    # Ensure response has all required fields
    result = {
        "question_offset": question_offset,
        "task": task_type,
        "question": example["question"],
        "true_answer": example["true_answer"],
        "output": response["output"],
        "intermediate_steps": response["intermediate_steps"],
        "is_correct": evaluation["is_correct"],
        "evaluation_score": evaluation["score"],
        "evaluation_feedback": evaluation["feedback"]
    }
    
    # Clear agent reference
    del gaia_agent
    
    return result

async def main():
    load_dotenv(override=True)
    eval_ds = load_dataset_by_type("small", "train")
    results = []
    
    args = parse_args()

    # Filter dataset for requested task type
    task_examples = [row for row in eval_ds if row["task"] == args.task_type]
    
    if not task_examples:
        raise ValueError(f"No examples found for task type: {args.task_type}")
        
    # Select examples to run
    if args.example_offsets is not None:
        examples_to_run = [(idx, task_examples[idx]) for idx in args.example_offsets]
    else:
        examples_to_run = [(idx, example) for idx, example in enumerate(task_examples[:args.num_examples])]
    
    # Run selected examples
    for question_offset, example in examples_to_run:
        result = await run_single_example(example, args.task_type, question_offset)
        results.append(result)
        print(json.dumps(result, indent=4))
    
    df = pd.DataFrame(results)
    df.to_json(f"results.json")

if __name__ == "__main__":
    asyncio.run(main()) 