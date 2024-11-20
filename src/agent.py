from openai import OpenAI
from agentjo import Agent as AgentJo
from src.tools.file_tools import file_inspect, file_visual_qa
from src.tools.python_tools import python_generate_and_run_tool
from src.tools.web_tools import wikipedia_search, web_search, web_visit_page
import time

def setup_llm():
    def llm(system_prompt: str, user_prompt: str) -> str:
        client = OpenAI()
        response = client.chat.completions.create(
            model='gpt-4o', # DO NOT CHANGE THIS
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    return llm

class Agent(AgentJo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_steps = []

    def run(self, task):
        """Run agent with enhanced intermediate step tracking"""
        # Assign the task properly
        self.assign_task(task)
        
        # Use the agent's built-in task handling
        outputs = super().run(task)
        
        # Get intermediate steps from agent's built-in tracking
        intermediate_steps = []
        for thought in self.thoughts:
            # Format the function call key to match subtasks_completed format
            function_name = thought.get('Equipped Function Name')
            function_inputs = thought.get('Equipped Function Inputs', {})
            
            # Build the key in the same format as subtasks_completed
            if function_inputs.get('instruction'):
                function_key = f'{function_name}(instruction="{function_inputs["instruction"]}")'
            else:
                params = [f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                         for k, v in function_inputs.items()]
                function_key = f'{function_name}({", ".join(params)})'
            
            step = {
                'Observation': thought.get('Observation'),
                'Thoughts': thought.get('Thoughts'),
                'Function Execution': {
                    'name': function_name,
                    'inputs': function_inputs,
                    'result': self.subtasks_completed.get(function_key, {})
                }
            }
            intermediate_steps.append(step)
        
        # Format final response
        result = {
            "question_offset": 0,
            "task": task,
            "question": task,
            "output": self.reply_user(task, stateful=False),  # Get final answer without adding to subtasks
            "intermediate_steps": intermediate_steps
        }
        
        return result

    def reply_user(self, user_input: str, stateful: bool = True) -> str:
        """Process user input and generate response"""
        task_type = self.shared_variables.get('task_type', '')
        
        # Handle GSM8K math problems
        if "GSM8K" in task_type:
            code_result = self.use_function(
                'python_generate_and_run_tool',
                {'instruction': user_input}
            )
            if isinstance(code_result, dict) and 'output_1' in code_result:
                result = code_result['output_1'].split('\n')[-1]
                return f"The answer is {result} minutes."
            return str(code_result)
        
        # Handle HotpotQA research questions
        elif "HotpotQA" in task_type:
            # First, try Wikipedia search
            wiki_result = self.use_function(
                'wikipedia_search',
                {'query': user_input}
            )
            
            # If Wikipedia doesn't yield enough info, try web search
            if not wiki_result or "No results found" in str(wiki_result):
                search_result = self.use_function(
                    'web_search',
                    {'query': user_input}
                )
                
                # Visit promising URLs from search results
                if search_result and isinstance(search_result, list):
                    for url in search_result[:2]:  # Visit top 2 results
                        page_content = self.use_function(
                            'web_visit_page',
                            {'url': url}
                        )
                        # Store in shared variables for context
                        self.shared_variables['task_state']['resources_used'].add(url)
        
        # Use default LLM response with accumulated context
        return super().reply_user(user_input, stateful=stateful)

def initialize(task_type="GSM8K"):
    """Initialize agent with appropriate configuration based on task type"""
    if task_type == "GSM8K":
        name = 'Mathematical Reasoning Expert'
        description = 'An expert agent specialized in solving mathematical word problems through careful step-by-step reasoning and computation'
        system_prompt = """
        1. For mathematical calculations:
            - Write Python code to compute values, especially for chained formulas
            - Show your work step-by-step
            - Verify results match intuitive estimates
        2. Always explain your reasoning process
        3. Double-check your work before providing final answer
        4. Examine the question answer phrasing and use that to express the answer as required.
        """
    elif task_type in ["HotpotQA-easy", "HotpotQA-medium", "HotpotQA-hard"]:
        name = 'Research Expert'
        description = 'An expert agent specialized in finding and synthesizing information from multiple sources'
        system_prompt = """You are an internet research expert. Follow these steps for each question:
1. Search Wikipedia first for relevant information
2. If needed, perform web searches for additional context
3. Visit specific pages to gather detailed information
4. Synthesize information from all sources to provide a complete, accurate answer
5. Always cite your sources in the response"""

    elif task_type == "GAIA":
        name = 'General AI Assistant'
        description = 'An expert agent specialized in solving real-world questions requiring multi-modal reasoning, web search, and tool use'
        system_prompt = """You are an advanced AI assistant focused on solving real-world questions that require multiple capabilities. Follow these steps:

1. Carefully analyze the question to identify required capabilities (reasoning, math, web search, etc.)
2. Break down complex problems into simpler sub-problems
3. For mathematical calculations:
   - Write Python code to compute values, especially for chained formulas
   - Show your work step-by-step
   - Verify results match intuitive estimates
4. For research questions:
   - Search authoritative sources
   - Cross-reference information
   - Synthesize findings into a clear answer
5. Always explain your reasoning process
6. Double-check your work before providing final answer
7. Examine the question answer phrasing and use that to express the answer as required.

Remember: Focus on accuracy and reliability over complexity. If a simple approach works, use it."""


    else:
        name = 'GaiaAgentJo'
        description = 'An agent that can solve various types of tasks using web search, reasoning, and mathematical calculations'
        system_prompt = "You are an expert at complex reasoning and internet research."

    base_agent = Agent(
        name=name,
        description=description,
        default_to_llm=False,
        shared_variables={
            'current_search': {
                'query': None,
                'results': [],
                'explored_urls': set(),
                'relevance_scores': {}
            },
            'task_state': {
                'current_step': None,
                'intermediate_results': {},
                'resources_used': set()
            },
            'task_type': task_type  # Add task_type to shared variables
        },
        global_context=system_prompt,  # Set the system prompt as global context
        llm=setup_llm()
    )

    # Create list of functions to assign
    functions = [
        file_inspect,
        file_visual_qa,
        python_generate_and_run_tool,
        wikipedia_search,
        web_search,
        web_visit_page
    ]

    # Assign functions
    base_agent = base_agent.assign_functions(functions)

    return base_agent