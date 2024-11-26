import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", "Importing LLMs from langchain.*")

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
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples to run per task type")
    parser.add_argument(
        "--example_offsets",
        type=lambda s: [int(x) for x in s.split(',')],
        help="Comma-separated list of example indices to run (e.g., '0,1,2')",
        default=[0]
    )
    parser.add_argument("--run_all_types", action="store_true", help="Run examples for all available task types")
    args = parser.parse_args()
    
    if args.run_all_types:
        args.task_type = None
    
    return args

async def run_single_example(example, task_type, question_offset):
    """Run a single example with a fresh agent instance and evaluate the response"""
    gaia_agent = initialize(task_type)
    evaluator_client = initialize_evaluator()
    
    # Configure system prompt based on task type
    if task_type == "GSM8K":
        system_prompt = "You are a mathematical reasoning expert. Break down problems step by step and use Python code to compute the final answer."
    elif task_type in ["HotpotQA-easy", "HotpotQA-medium"]:
        system_prompt = "You are an internet research expert. Find and synthesize information from multiple sources to answer questions accurately."
    else:
        system_prompt = "You are an expert at complex reasoning and internet research."
        
    gaia_agent.global_context = system_prompt
    
    # Run the example and get full response including intermediate steps
    response = gaia_agent.run(example["question"])
    
    # Evaluate the response 
    evaluation = await evaluate_response(evaluator_client, response, example)
    # LEAVE COMMENTED
    # evaluation = {
    #     "is_correct": None,
    #     "score": -1,
    #     "feedback": ""
    # }
    
    # Clean intermediate steps to ensure JSON serialization
    cleaned_steps = []
    for step in response["intermediate_steps"]:
        cleaned_step = {}
        for key, value in step.items():
            if isinstance(value, dict):
                cleaned_value = {k: v for k, v in value.items() 
                               if isinstance(v, (str, int, float, bool, list, dict))}
                cleaned_step[key] = cleaned_value
            else:
                cleaned_step[key] = value
        cleaned_steps.append(cleaned_step)
    
    # Create result with cleaned data
    result = {
        "question_offset": question_offset,
        "task": task_type,
        "question": example["question"],
        "true_answer": example["true_answer"],
        "output": response["output"],
        "intermediate_steps": cleaned_steps,
        "is_correct": evaluation["is_correct"],
        "evaluation_score": evaluation["score"],
        "evaluation_feedback": evaluation["feedback"]
    }
    
    # Clear agent reference
    del gaia_agent
    
    return result

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def clean_for_serialization(obj, skip_keys={'shared_variables', 'kwargs', 'agent'}):
    """Recursively clean dictionary by removing problematic keys and converting sets."""
    if isinstance(obj, dict):
        return {
            k: clean_for_serialization(v, skip_keys)
            for k, v in obj.items()
            if k not in skip_keys
        }
    elif isinstance(obj, list):
        return [clean_for_serialization(item, skip_keys) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    return obj

async def main():
    load_dotenv(override=True)
    eval_ds = load_dataset_by_type("small", "train")
    results = []
    
    args = parse_args()

    # Get unique task types from dataset
    all_task_types = list(set(row["task"] for row in eval_ds))
    
    # Determine which task types to run
    task_types_to_run = all_task_types if args.run_all_types else [args.task_type]
    if not args.run_all_types and not args.task_type:
        raise ValueError("Must specify either --task_type or --run_all_types")

    # Process each task type
    for task_type in task_types_to_run:
        task_examples = [row for row in eval_ds if row["task"] == task_type]
        
        if not task_examples:
            print(f"Warning: No examples found for task type: {task_type}")
            continue
            
        # Select examples to run
        if args.example_offsets is not None:
            examples_to_run = [(idx, task_examples[idx]) for idx in args.example_offsets]
        else:
            examples_to_run = [(idx, example) for idx, example in enumerate(task_examples[:args.num_examples])]
        
        print(f"\nProcessing {len(examples_to_run)} examples for task type: {task_type}")
        
        # Run selected examples
        for question_offset, example in examples_to_run:
            result = await run_single_example(example, task_type, question_offset)
            try:
                cleaned_result = clean_for_serialization(result)
                print(json.dumps(cleaned_result, indent=4))
            except TypeError as e:
                print(f"Warning: Could not serialize result: {e}")
            results.append(cleaned_result)

    # Save results
    try:
        with open("results.jsonl", "w") as f:
            for result in results:
                json.dump(result, f)
                f.write('\n')
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 