from openai import OpenAI
from agentjo import Agent
from agentjo_package.wrapper import ConversationWrapper
from src.tools.file_tools import file_inspect, file_visual_qa
from src.tools.python_tools import python_generate_and_run_tool
from src.tools.web_tools import wikipedia_search
from functools import wraps
import inspect
import time

def wrap_for_agent(func):
    """Decorator that wraps a function to make it compatible with AgentJo's pattern and captures execution details"""
    if not inspect.isfunction(func) and not inspect.iscoroutinefunction(func):
        raise ValueError(f"{func} must be a function or coroutine function")
        
    @wraps(func)    
    def wrapper(shared_variables, *args, **kwargs):
        '''Executes {func.__name__} with provided arguments and captures execution details'''
        # Execute function
        start_time = time.time()
        
        # Add shared_variables to kwargs
        kwargs['shared_variables'] = shared_variables
        result = func(*args, **kwargs)
        
        end_time = time.time()
        
        # Capture execution details in shared_variables
        if 'function_executions' not in shared_variables:
            shared_variables['function_executions'] = []
            
        execution_record = {
            'function': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'result': result,
            'start_time': start_time,
            'end_time': end_time,
            'time_duration': end_time - start_time
        }
        shared_variables['function_executions'].append(execution_record)
        
        return result
    
    # Copy attributes
    wrapper.__module__ = func.__module__
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__annotations__ = func.__annotations__
    wrapper.__defaults__ = func.__defaults__
    wrapper.__dict__.update(func.__dict__)
    
    return wrapper

def setup_llm():
    def llm(system_prompt: str, user_prompt: str) -> str:
        client = OpenAI()
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    return llm

class Agent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intermediate_steps = []

    def _capture_intermediate_steps(self):
        """Convert thoughts and execution details into structured intermediate steps"""
        steps = []
        
        # Iterate over the conversation and function executions
        for i, (conversation, execution) in enumerate(zip(self.shared_variables.get('Conversation', []), self.shared_variables.get('function_executions', []))):
            step_entry = {
                'Conversation': conversation,
                'Function Execution': {
                    'name': execution['function'],
                    'args': execution['args'],
                    'kwargs': execution['kwargs'],
                    'result': execution['result'],
                    'time_duration': execution['time_duration']
                }
            }
            steps.append(step_entry)
        
        self.intermediate_steps = steps
        return steps

    def run(self, task):
        """Run agent with enhanced intermediate step tracking"""
        self.conversation = ConversationWrapper(
            agent=self,
            person="User",
            verbose=True,
        )
        
        # Start the conversation with the task
        result = self.conversation.chat(task)
        
        # Capture intermediate steps
        intermediate_steps = self._capture_intermediate_steps()
        
        # Format final response with intermediate steps
        response = {
            "question_offset": 0,  # This could be passed in as a parameter if needed
            "task": task,
            "question": task,
            "output": self._format_final_answer(task),
            "intermediate_steps": intermediate_steps
        }
        
        return response

    def use_function(self, function_name: str, function_params: dict, subtask: str = '', stateful: bool = True):
        """Override use_function to capture execution details"""
        result = super().use_function(function_name, function_params, subtask, stateful)
        
        # Record the execution
        if 'function_executions' not in self.shared_variables:
            self.shared_variables['function_executions'] = []
            
        execution_record = {
            'function': function_name,
            'args': [],
            'kwargs': function_params,
            'result': result,
            'start_time': time.time(),
            'end_time': time.time(),
            'time_duration': 0  # Placeholder, update with actual time duration
        }
        self.shared_variables['function_executions'].append(execution_record)
        
        return result

    def _format_final_answer(self, original_task):
        """Reformulate final answer using conversation history"""
        # Get the full conversation history
        conversation_history = self.conversation.shared_variables['Conversation']
        conversation_summary = self.conversation.shared_variables['Summary of Conversation']
        
        reformulation_prompt = f"""Earlier you were asked the following:

{original_task}

Here is the conversation and reasoning process:
Conversation History: {conversation_history}
Conversation Summary: {conversation_summary}

Provide a clear, concise FINAL ANSWER that:
1. Directly addresses the original question
2. Is as concise as possible while being complete
3. Uses numerical values where appropriate
4. Follows any formatting instructions in the original question
5. Draws from the key insights in the conversation

Format: FINAL ANSWER: [your answer]"""

        final_response = self.llm(
            system_prompt="You are a precise and direct assistant that provides clear final answers.",
            user_prompt=reformulation_prompt
        )
        
        return final_response.split("FINAL ANSWER: ")[-1].strip()

def initialize():
    base_agent = Agent(
        name='GaiaAgentJo',
        description='An agent that can solve various types of tasks using web search, reasoning, and mathematical calculations',
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
            'function_executions': []  # Store function execution details
        },
        llm=setup_llm()
    )

    # Create properly wrapped functions
    wrapped_functions = [
        wrap_for_agent(file_inspect),
        wrap_for_agent(file_visual_qa),
        wrap_for_agent(python_generate_and_run_tool),
        wrap_for_agent(wikipedia_search)
    ]

    # Assign wrapped functions
    base_agent = base_agent.assign_functions(wrapped_functions)

    return base_agent