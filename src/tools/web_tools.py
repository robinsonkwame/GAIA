import time
from typing import Optional
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from functools import wraps
from typing import Callable, Any
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict

# Load environment variables at module level
load_dotenv(override=True)

def rate_limit(limit: float) -> Callable:
    """Decorator to rate limit function calls
    
    Args:
        limit: Minimum time in seconds between function calls
    """
    last_call = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get current time
            current_time = time.time()
            
            # Check if enough time has passed since last call
            if func.__name__ in last_call:
                elapsed = current_time - last_call[func.__name__]
                if elapsed < limit:
                    time.sleep(limit - elapsed)
            
            # Update last call time
            last_call[func.__name__] = time.time()
            
            # Extract shared_variables if present
            shared_variables = kwargs.pop('shared_variables', None)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(1.0)  # 1 second rate limit
def wikipedia_search(query: str, num_results: Optional[int] = 1, shared_variables: dict = None) -> str:
    """Searches Wikipedia for the given query and returns relevant article content.
    
    Args:
        query: The search query string to look up on Wikipedia
        num_results: Optional number of results to return (default: 2)
        shared_variables: Optional shared variables dict
        
    Returns:
        str: Combined content from relevant Wikipedia articles
    """
    try:
        wiki = WikipediaAPIWrapper(
            top_k_results=num_results,
            doc_content_chars_max=2000  # Limit content length per article
        )
        tool = WikipediaQueryRun(api_wrapper=wiki)
        results = tool.run(query)
        return results
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

@rate_limit(1.0)  # 1 second rate limit
def web_search(query: str, shared_variables: dict = None) -> List[Dict]:
    """Searches the web using Google Custom Search API.
    
    Args:
        query: The search query string
        shared_variables: Optional shared variables dict
        
    Returns:
        list: List of search results with title, snippet and URL
    """
    try:
        api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_KEY')
        cx = os.getenv('GOOGLE_CUSTOM_SEARCH_KEY_CX')
        
        if not api_key or not cx:
            return [{"error": "Google Custom Search API credentials not configured"}]
            
        # Send request to Google Custom Search API
        response = requests.get(
            f'https://www.googleapis.com/customsearch/v1',
            params={'key': api_key, 'cx': cx, 'q': query}
        )
        
        # Extract search results
        data = response.json()
        results = []
        
        items = data.get('items', [])
        for item in items:
            result = {
                'title': item.get('title'),
                'snippet': item.get('snippet'),
                'url': item.get('link')
            }
            results.append(result)
            
        return results

    except Exception as e:
        return [{"error": f"Error searching the web: {str(e)}"}]

def web_visit_page(url: str) -> str:
    """Visits a webpage and extracts its main content.
    
    Args:
        url: The URL of the webpage to visit
        
    Returns:
        str: The extracted main content of the webpage
    """
    pass