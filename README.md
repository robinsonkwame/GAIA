# Beating GAIA with AgentJo 🚀


### How to run tests?

First, install requirements:
```bash
pip install -r requirements.txt
```

Setup your secrets in a `.env`file:
```bash
HUGGINGFACEHUB_API_TOKEN
SERPAPI_API_KEY
OPENAI_API_KEY
ANTHROPIC_API_KEY
```

And optionally if you want to use Anthropic models via AWS bedrock:
```bash
AWS_BEDROCK_ID
AWS_BEDROCK_KEY
```

Then run `python -m src/main.py` to launch tests!