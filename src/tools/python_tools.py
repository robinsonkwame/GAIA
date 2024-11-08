from langchain_experimental.utilities import PythonREPL

def python_generator_tool(shared_variables, instruction: str) -> dict:
    """Generates Python code based on natural language instruction.
    
    Args:
        shared_variables: Dictionary containing shared agent state
        instruction: Natural language instruction for code generation
        
    Returns:
        dict: Contains generated code under 'Generated Code' key
    """
    agent = shared_variables['agent']
    
    response = agent.llm(
        system_prompt="""You are a Python code generator. Generate only executable Python code based on the instruction.
        Only use allowed imports: math, numpy, random, datetime, re, pandas.
        Do not include markdown code fences or language indicators.
        Do not define new functions. Ensure all required output uses print().
        Use equipped functions whenever possible.""",
        user_prompt=instruction
    )
    
    code = response.strip('`').replace('python\n', '').strip()
    return {"Generated Code": code}

def python_run_tool(shared_variables, code_snippet: str) -> str:
    """Runs Python code snippet using LangChain's PythonREPL.
    
    Args:
        shared_variables: Dictionary containing shared agent state
        code_snippet: Python code to execute
        
    Returns:
        str: Output from code execution or error message
    """
    repl = PythonREPL()
    
    setup_code = """
import math
import numpy
import random
import datetime
import re
import pandas
"""
    
    full_code = setup_code + "\n" + code_snippet
    
    try:
        result = repl.run(full_code)
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def python_generate_and_run_tool(shared_variables, instruction: str) -> str:
    """Generates and executes Python code based on natural language instruction.
    
    Args:
        shared_variables: Dictionary containing shared agent state
        instruction: Natural language instruction for code generation
        
    Returns:
        str: Combined output containing both generated code and execution results
    """
    generated = python_generator_tool(shared_variables, instruction)
    code = generated["Generated Code"]
    result = python_run_tool(shared_variables, code)
    return f"Generated Code:\n{code}\n\nExecution Output:\n{result}" 