from openai import AsyncOpenAI
from agentjo import AsyncAgent
from .tools.python_tools import python_generate_and_run_tool
from .tools.web_tools import web_search, web_visit_page
from .tools.file_tools import file_inspect, file_visual_qa

async def setup_llm():
    async def llm_async(system_prompt: str, user_prompt: str) -> str:
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model='gpt-4o-mini',
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content

    return llm_async

async def initialize():
    return AsyncAgent(
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
        llm=await setup_llm()
    ).assign_functions([
        web_search,
        web_visit_page, 
        file_inspect,
        file_visual_qa,
        python_generate_and_run_tool
    ]) 