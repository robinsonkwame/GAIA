# Beating GAIA with AgentJo 🚀


### How to run tests?

First, install requirements:
```bash
pip install -r requirements.txt
```

Setup your secrets in a `.env`file:
```bash
HUGGINGFACEHUB_API_TOKEN= ...
SERPAPI_API_KEY= ...
OPENAI_API_KEY= ...
ANTHROPIC_API_KEY= ...

AWS_BEDROCK_ACCESS_KEY= ...
AWS_BEDROCK_SECRET_KEY= ...

AZURE_OPENAI_KEY= ...
AZURE_OPENAI_ENDPOINT= ...
API_VERSION= ...

GOOGLE_CUSTOM_SEARCH_KEY= ...
GOOGLE_CUSTOM_SEARCH_KEY_CX= ...
```

#### Running Tests

The main script supports several command-line arguments to control which tests to run:

1. **Run a specific task type**:
```bash
python -m src.main --task_type GSM8K
python -m src.main --task_type HotpotQA-easy
python -m src.main --task_type HotpotQA-medium
python -m src.main --task_type HotpotQA-hard
python -m src.main --task_type GAIA
```

2. **Control number of examples**:
```bash
# Run first 5 examples of GSM8K
python -m src.main --task_type GSM8K --num_examples 5
```

3. **Run specific examples by offset**:
```bash
# Run task examples at offset 0, 3, and 7
python -m src.main --task_type GSM8K --example_offsets 0,3,7
```

4. **Default behavior**:
- Without arguments, runs 1 example from GSM8K task
```bash
python -m src.main
```

#### Test Results

All test results are automatically saved to `results.json`. Each result includes:
- Question offset (index)
- Task type
- Original question and true answer
- Agent's output and intermediate reasoning steps
- Evaluation metrics:
  - Correctness flag
  - Numerical score
  - Detailed feedback

The results are also printed to the console in real-time as JSON for immediate review.

Example result structure:
```json
{
    "question_offset": 0,
    "task": "GSM8K",
    "question": "...",
    "true_answer": "...",
    "output": "...",
    "intermediate_steps": [...],
    "is_correct": true,
    "evaluation_score": 1.0,
    "evaluation_feedback": "..."
}
```