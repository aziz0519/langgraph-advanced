import requests
import yfinance as yf
from pprint import pformat
from typing import Callable
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig, ActionRequest


@tool("lookup_stock")
def lookup_stock_symbol(company_name: str) -> str:
    """
    Converts a company name to its stock symbol using a financial API.

    Parameters:
        company_name (str): The full company name (e.g., 'Tesla').

    Returns:
        str: The stock symbol (e.g., 'TSLA') or an error message.
    """
    api_url = "https://www.alphavantage.co/query"
    params = {
        "function": "SYMBOL_SEARCH",
        "keywords": company_name,
        "apikey": "your_alphavantage_api_key"
    }
    
    response = requests.get(api_url, params=params)
    data = response.json()
    
    if "bestMatches" in data and data["bestMatches"]:
        return data["bestMatches"][0]["1. symbol"]
    else:
        return f"Symbol not found for {company_name}."


@tool("fetch_stock_data")
def fetch_stock_data_raw(stock_symbol: str) -> dict:
    """
    Fetches comprehensive stock data for a given symbol and returns it as a combined dictionary.

    Parameters:
        stock_symbol (str): The stock ticker symbol (e.g., 'TSLA').
        period (str): The period to analyze (e.g., '1mo', '3mo', '1y').

    Returns:
        dict: A dictionary combining general stock info and historical market data.
    """
    period = "1mo"
    try:
        # DEBUG: Log environment info
        print(f"DEBUG: Fetching stock data for {stock_symbol}")
        print(f"DEBUG: yfinance version: {yf.__version__}")
        
        stock = yf.Ticker(stock_symbol)

        # Retrieve general stock info and historical market data
        stock_info = stock.info  # Basic company and stock data
        print(f"DEBUG: Successfully retrieved stock info")
        
        stock_history = stock.history(period=period).to_dict()  # Historical OHLCV data
        print(f"DEBUG: Successfully retrieved stock history")

        # Combine both into a single dictionary
        combined_data = {
            "stock_symbol": stock_symbol,
            "info": stock_info,
            "history": stock_history
        }

        return pformat(combined_data)

    except Exception as e:
        return {"error": f"Error fetching stock data for {stock_symbol}: {str(e)}"}


@tool
def place_order(
    symbol: str,
    action: str,
    shares: int,
    limit_price: float,
    order_type: str = "limit",
) -> dict:
    """
    Execute a stock order.

    Parameters:
    - symbol: Ticker
    - action: "buy" or "sell"
    - shares: Number of shares to trade (pre-computed by the agent)
    - limit_price: Limit price per share
    - order_type: Order type, default "limit"

    Returns:
    - status: Execution result (simulated)
    - symbol
    - shares
    - limit_price
    - total_spent
    - type: Order type used
    - action
    """
    total_spent = round(int(shares) * limit_price, 2)
    return {
        "status": "filled",
        "symbol": symbol,
        "shares": int(shares),
        "limit_price": limit_price,
        "total_spent": total_spent,
        "type": order_type,
        "action": action,
    }


@tool
def ask_question(question: str) -> str:
    """
    Asks a human a question and waits for their response using Human-in-the-Loop (HITL).
    
    This tool interrupts the agent's execution to collect human input, then resumes
    with the human's answer. Use this when you need clarification, approval, or
    information that only the human can provide.
    
    Parameters:
        question (str): The question to ask the human. Be specific and clear.
        
    Returns:
        str: The human's response to the question.
        
    Example:
        >>> ask_question("What is your preferred investment budget for this trade?")
        # Agent pauses here, waiting for human input
        # Returns: "$5000" (or whatever the human responds)
    """
    
    # Create standardized interrupt request
    request = HumanInterrupt(
        action_request=ActionRequest(
            action="ask_question",  # The tool name
            args={"question": question}  # The tool arguments
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,      # User can skip the question
            allow_respond=True,     # User can provide text response
            allow_edit=False,       # User cannot edit the question
            allow_accept=False      # No accept action for questions
        ),
        description=f"Please answer the following question: {question}"
    )
    
    # Send interrupt and wait for response
    response = interrupt(request)[0]
    
    # LangGraph Studio returns a list of responses - extract the first one
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    
    # Handle different response types
    if response["type"] == "response":
        # User provided text response
        return response["args"]  # This is the string response
    elif response["type"] == "ignore":
        # User chose to skip
        return "User chose not to answer this question."
    else:
        # Unexpected response type
        return f'Unexpected response type: {response["type"]}'


def add_approval(main_tool: Callable | BaseTool, allow_edit: bool = False) -> BaseTool:
    """
    Wrap a tool to support human-in-the-loop review using standardized format.
    
    Args:
        main_tool: The tool to wrap with HITL approval
        allow_edit: Whether to allow editing tool arguments (default: False)
        
    Note on allow_edit:
        - False (recommended): User can only approve or reject. Safer for production.
          If user wants changes, they reject and instruct the agent differently.
        - True (educational): User can edit arguments, but this can break agent reasoning
          if they change the action completely (e.g., Tesla → Microsoft).
          Use with caution and consider validation in production.
    """
    if not isinstance(main_tool, BaseTool):
        main_tool = tool(main_tool)

    @tool(  
        main_tool.name,
        description=main_tool.description,
        args_schema=main_tool.args_schema
    )
    def call_main_tool_with_hitl(config: RunnableConfig, **tool_input):
        # Create standardized interrupt request
        request = HumanInterrupt(
            action_request=ActionRequest(
                action=main_tool.name,  # The tool being called
                args=tool_input         # The tool arguments
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,      # User can skip/reject the tool
                allow_respond=False,    # No text response needed
                allow_edit=allow_edit,  # Configurable: allow editing args
                allow_accept=True       # User can approve the tool
            ),
            description=f"Please review and approve the '{main_tool.name}' tool call"
        )
        
        # Send interrupt and wait for response
        response = interrupt(request)[0]
        
        # Handle different response types
        if response["type"] == "accept":
            # Tool approved - execute with original args
            return main_tool.invoke(tool_input, config)
        
        elif response["type"] == "edit":
            # Tool approved with edited args
            # WARNING: Edited args may break agent's reasoning if changed significantly
            edited_action: ActionRequest = response["args"]
            return main_tool.invoke(edited_action.args, config)
        
        elif response["type"] == "ignore":
            # Tool rejected
            return "Cancelled by human. Continue without executing that tool and provide next steps."
        
        else:
            # Unexpected response type
            return f'Unexpected response type: {response["type"]}. Tool not executed.'
        

    return call_main_tool_with_hitl


system_message = """
You are a financial advisor assistant. Use the provided tools to ground your answers
in up-to-date market data. Be concise, factual, and risk-aware.

Be decisive: when you have sufficient information to act, proceed with tool calls without
asking for confirmation. Only if information is missing or uncertain, ask a concise 
clarifying question.

When preparing or describing actions, include appropriate parameters (e.g., symbol, shares,
limit price, budgets) based on available data. Do not fabricate numbers or facts.
"""


graph = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[lookup_stock_symbol, fetch_stock_data_raw, add_approval(place_order), ask_question],
    prompt=system_message,
)
