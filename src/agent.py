from openai import OpenAI
from agentjo import Agent, ConversationWrapper
from functools import wraps
import inspect
import asyncio
from .tools.python_tools import python_generate_and_run_tool
from .tools.web_tools import web_search, web_visit_page
from .tools.file_tools import file_inspect, file_visual_qa

def wrap_for_agent(func):
    """Decorator that wraps a function to make it compatible with AgentJo's pattern"""
    if not inspect.isfunction(func) and not inspect.iscoroutinefunction(func):
        raise ValueError(f"{func} must be a function or coroutine function")
        
    @wraps(func)    
    def wrapper(shared_variables, *args, **kwargs):
        '''Executes {func.__name__} with provided arguments'''
        return func(shared_variables, *args, **kwargs)
    
    # Copy attributes
    wrapper.__module__ = func.__module__
    wrapper.__name__ = func.__name__
    wrapper.__qualname__ = func.__qualname__
    wrapper.__annotations__ = func.__annotations__
    wrapper.__defaults__ = func.__defaults__
    wrapper.__dict__.update(func.__dict__)
    
    if hasattr(func, '__annotations__'):
        wrapper.__annotations__ = func.__annotations__

    wrapper.__get__ = lambda self, obj, objtype=None: self
    
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
            }
        },
        llm=setup_llm()
    )

    # Create properly wrapped functions
    wrapped_functions = [
        wrap_for_agent(web_search),
        wrap_for_agent(web_visit_page),
        wrap_for_agent(file_inspect),
        wrap_for_agent(file_visual_qa),
        wrap_for_agent(python_generate_and_run_tool)
    ]

    # Assign wrapped functions
    base_agent = base_agent.assign_functions(wrapped_functions)

    # Create conversation wrapper
    conversation_agent = ConversationWrapper(
        base_agent,
        persistent_memory={}
    )

    return conversation_agent