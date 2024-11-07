import json
import asyncio
import pandas as pd
from dotenv import load_dotenv

from .utils import load_dataset_by_type
from .agent import initialize

async def main():
    load_dotenv(override=True)
    eval_ds = load_dataset_by_type("small", "train")
    gaia_agent = await initialize()
    results = []
    
    for row in eval_ds:
        question = row["question"]
        task_type = row["task"]
        
        # Configure context based on task type
        if task_type in ["GSM8K"]:
            system_prompt = "You are a mathematical reasoning expert."
        elif task_type in ["HotpotQA-easy", "HotpotQA-medium"]:
            system_prompt = "You are a web research expert."
        else:
            system_prompt = "You are an expert at complex reasoning and research."
            
        gaia_agent.global_context = system_prompt
        result = await gaia_agent.run(question)
        
        results.append({
            "output": str(result),
            "intermediate_steps": gaia_agent.shared_variables['task_state']['intermediate_results']
        })
        print(json.dumps(results[-1], indent=4))
        break
    
    df = pd.DataFrame(results)
    df.to_json(f"results.json")

if __name__ == "__main__":
    asyncio.run(main()) 