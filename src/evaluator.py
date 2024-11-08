import os
from typing import Dict
from openai import OpenAI
from scripts.evaluation.evaluation import extract_numbers
import asyncio

def initialize_evaluator():
    """Initialize the OpenAI client"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI()

def evaluate_with_llm(client: OpenAI, instruction: str, response: str, reference_answer: str) -> Dict:
    """Evaluate using OpenAI's API"""
    system_prompt = """You are an expert evaluator. For math problems:
1. Check if the reasoning is clear and logical
2. Verify if the calculation steps are correct
3. Confirm the final numerical answer matches the reference
4. Consider a response correct only if both reasoning and final answer are correct"""

    user_prompt = f"""Question: {instruction}

Response to evaluate: {response}

Reference answer: {reference_answer}

Evaluate the response for both reasoning and correctness. First provide specific feedback about the reasoning and calculation steps, then on a new line after [RESULT], provide a score of:
1 - if both reasoning and final answer are correct
0 - if either reasoning is unclear/incorrect or the final answer is wrong"""

    eval_result = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return eval_result.choices[0].message.content

async def evaluate_response(client: OpenAI, response: Dict, example: Dict) -> Dict:
    """Evaluate a single response against the reference answer"""
    # Extract numerical answer from response
    response_numbers = extract_numbers(response["output"])
    final_number = response_numbers[-1] if response_numbers else None
    
    # Get reference answer
    if isinstance(example["true_answer"], str) and "####" in example["true_answer"]:
        # Handle GSM8K format where answer is after ####
        ref_parts = example["true_answer"].split("####")
        ref_reasoning = ref_parts[0].strip()
        ref_number = float(ref_parts[1].strip().replace(",", ""))
    else:
        ref_reasoning = ""
        ref_number = float(example["true_answer"]) if example["true_answer"] else None

    # Use LLM evaluation for reasoning and correctness
    eval_result = evaluate_with_llm(
        client,
        example["question"],
        response["output"],
        example["true_answer"]
    )
    
    try:
        feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
        score = int(score)
        
        # Additional numerical verification for math problems
        if final_number is not None and ref_number is not None:
            numerical_match = abs(final_number - ref_number) < 0.01
            if not numerical_match:
                score = 0
                feedback += "\nNumerical answer mismatch."
    except:
        feedback = "Error parsing evaluation"
        score = 0
    
    return {
        "is_correct": score == 1,
        "score": score,
        "feedback": feedback,
        "extracted_answer": final_number,
        "reference_answer": ref_number
    } 

def test_evaluator():
    """Test function to verify the evaluator is working correctly"""
    # Initialize the evaluator
    client = initialize_evaluator()
    
    # Test example
    test_example = {
        "question": "Bob is tilling a plot of his garden. The plot is 110 feet wide by 120 feet long. His tiller digs a swath two feet wide, and he can till 1 foot of ground in about 2 seconds. How long will it take him to till this plot of land, in minutes?",
        "true_answer": "If Bob goes along the side that's 120 feet long, he will till 110 / 2 = 55 rows. Each of these rows are 120 feet long, so he will push the tiller a total of 120 * 55 = <<120*55=6600>>6,600 feet. He tills 1 linear foot of ground every 2 seconds, so it will take him 2 * 6,600 = 13,200 seconds to till this plot 13,200 seconds is 13,2000 / 60 = <<13200/60=220>>220 minutes #### 220"
    }
    
    # Test responses - one correct and one incorrect
    test_responses = [
        {
            "output": "Let me solve this step by step:\n1. Plot dimensions: 110 feet wide × 120 feet long\n2. Tiller width: 2 feet\n3. Number of rows needed: 110 ÷ 2 = 55 rows\n4. Length of each row: 120 feet\n5. Total distance to till: 55 rows × 120 feet = 6,600 feet\n6. Time per foot: 2 seconds\n7. Total seconds: 6,600 feet × 2 seconds = 13,200 seconds\n8. Convert to minutes: 13,200 ÷ 60 = 220 minutes\nTherefore, it will take Bob 220 minutes to till the entire plot.",
            "description": "Correct answer with clear reasoning"
        },
        {
            "output": "The plot is 110 × 120 feet, so that's 13,200 square feet. At 2 seconds per foot, it will take 26,400 seconds or 440 minutes.",
            "description": "Incorrect answer - wrong calculation method"
        }
    ]
    
    expected_results = [True, False]  # Expected results for correct and incorrect responses
    
    print("Running evaluator test...\n")
    
    # Test each response
    for i, (test_response, expected_correct) in enumerate(zip(test_responses, expected_results), 1):
        print(f"\nTesting response {i} ({test_response['description']}):")
        print("-" * 50)
        print(f"Response: {test_response['output']}\n")
        
        # Run evaluation
        evaluation = asyncio.run(evaluate_response(client, test_response, test_example))
        
        # Assert the evaluation result matches expected
        assert evaluation['is_correct'] == expected_correct, \
            f"Test {i} failed: Expected is_correct={expected_correct}, got {evaluation['is_correct']}"
        
        print(f"Evaluation results:")
        print(f"Score: {evaluation['score']}")
        print(f"Is correct: {evaluation['is_correct']}")
        print(f"Extracted answer: {evaluation['extracted_answer']}")
        print(f"Reference answer: {evaluation['reference_answer']}")
        print(f"Feedback: {evaluation['feedback']}")
        print("-" * 50)
    
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_evaluator()