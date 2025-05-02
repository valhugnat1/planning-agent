from my_agent.utils.calendar_tools import  get_calendar_tool, create_calendar_tool
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=3), get_calendar_tool, create_calendar_tool]