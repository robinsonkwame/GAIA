import time
from typing import Optional, Tuple
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from functools import wraps
from typing import Callable, Any
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict
from PyPDF2 import PdfReader
from io import BytesIO
from markdown import markdown as md

# Load environment variables at module level
load_dotenv(override=True)

class WebBrowser:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        self.current_page = None
        self.viewport_position = 0
        self.page_content = ""
        
    def visit_page(self, url: str) -> Tuple[str, str]:
        """Visit a webpage and return its content"""
        try:
            response = requests.get(url, headers={"User-Agent": self.user_agent}, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type:
                reader = PdfReader(BytesIO(response.content))
                content = "\n".join(page.extract_text() for page in reader.pages)
            else:
                content = md(response.text)
            
            self.current_page = url
            self.page_content = content
            self.viewport_position = 0
            
            header = f"Address: {url}\n"
            return header, content
            
        except Exception as e:
            return f"Error visiting page: {str(e)}", ""

    def page_up(self) -> Tuple[str, str]:
        """Scroll viewport up"""
        if not self.current_page:
            return "No page loaded", ""
        
        viewport_size = 1024 * 5
        self.viewport_position = max(0, self.viewport_position - viewport_size)
        visible_content = self.page_content[self.viewport_position:self.viewport_position + viewport_size]
        
        header = f"Address: {self.current_page}\nViewport position: {self.viewport_position}"
        return header, visible_content

    def page_down(self) -> Tuple[str, str]:
        """Scroll viewport down"""
        if not self.current_page:
            return "No page loaded", ""
            
        viewport_size = 1024 * 5
        self.viewport_position = min(len(self.page_content) - viewport_size, 
                                   self.viewport_position + viewport_size)
        visible_content = self.page_content[self.viewport_position:self.viewport_position + viewport_size]
        
        header = f"Address: {self.current_page}\nViewport position: {self.viewport_position}"
        return header, visible_content

    def jump_to_bottom(self) -> Tuple[str, str]:
        """Jump to bottom of page"""
        if not self.current_page:
            return "No page loaded", ""
            
        viewport_size = 1024 * 5
        self.viewport_position = max(0, len(self.page_content) - viewport_size)
        visible_content = self.page_content[self.viewport_position:]
        
        header = f"Address: {self.current_page}\nViewport position: {self.viewport_position} (bottom)"
        return header, visible_content

# Initialize browser instance
browser = WebBrowser()

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

@rate_limit(1.0)
def web_visit_page(url: str, shared_variables: dict = None) -> str:
    """Visits a webpage and extracts its main content.
    
    Args:
        url: The URL of the webpage to visit
        shared_variables: Optional shared variables dict
        
    Returns:
        str: The extracted main content of the webpage
    """
    header, content = browser.visit_page(url)
    return header + "\n=======================\n" + content

@rate_limit(1.0)
def web_page_up(shared_variables: dict = None) -> str:
    """Scrolls the viewport up one page length.
    
    Args:
        shared_variables: Optional shared variables dict
        
    Returns:
        str: The new viewport content
    """
    header, content = browser.page_up()
    return header + "\n=======================\n" + content

@rate_limit(1.0) 
def web_page_down(shared_variables: dict = None) -> str:
    """Scrolls the viewport down one page length.
    
    Args:
        shared_variables: Optional shared variables dict
        
    Returns:
        str: The new viewport content
    """
    header, content = browser.page_down()
    return header + "\n=======================\n" + content

@rate_limit(1.0)
def web_jump_to_bottom(shared_variables: dict = None) -> str:
    """Jumps to the bottom of the current webpage.
    
    Args:
        shared_variables: Optional shared variables dict
        
    Returns:
        str: The new viewport content
    """
    header, content = browser.jump_to_bottom()
    return header + "\n=======================\n" + content

def test_web_tools():
    """Test web browsing functionality"""
    print("Testing web tools...")
    
    # Visit Wikipedia page
    print("\n1. Visiting Wikipedia page...")
    result = web_visit_page("https://en.wikipedia.org/wiki/Python_(programming_language)")
    print(result[:500] + "...\n")
    
    # Jump to bottom
    print("2. Jumping to bottom of page...")
    result = web_jump_to_bottom()
    print(result[:500] + "...\n")
    
    # Search for word "Text is available"
    print("3. Searching for 'Text is available'...")
    content = browser.page_content.lower()
    position = content.find("Text is available".lower())
    if position != -1:
        # Get context around the word
        start = max(0, position - 100)
        end = min(len(content), position + 100)
        context = content[start:end]
        print(f"Found 'Text is available' in context:\n{context}\n")
    else:
        print("Word 'Text is available' not found in content")

if __name__ == "__main__":
    test_web_tools()