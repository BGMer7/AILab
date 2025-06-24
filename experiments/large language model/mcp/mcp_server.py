from fastmcp import FastMCP

mcp = FastMCP(name="MyAssistantServer")


@mcp.tool()
def add(a: float, b: float) -> float:
    """加法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之和
    """
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """减法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之差 (a - b)
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """乘法运算

    参数:
    a: 第一个数字
    b: 第二个数字

    返回:
    两数之积
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """除法运算

    参数:
    a: 被除数
    b: 除数

    返回:
    两数之商 (a / b)

    异常:
    ValueError: 当除数为零时
    """
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

@mcp.tool()
def get_time() -> str:
    """获取当前时间

    返回:
    当前时间字符串 (格式: YYYY-MM-DD HH:MM:SS)
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Static resource
@mcp.resource("config://version")
def get_version(): 
    return "2.0.1"

# Dynamic resource template
@mcp.resource("users://{user_id}/profile")
def get_profile(user_id: int):
    # Fetch profile for user_id...
    return {"name": f"User {user_id}", "status": "active"}

# Basic dynamic resource returning a string
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    """Provides a simple greeting message."""
    return "Hello from FastMCP Resources!"

# Resource returning JSON data (dict is auto-serialized)
@mcp.resource("data://config")
def get_config() -> dict:
    """Provides application configuration as JSON."""
    return {
        "theme": "dark",
        "version": "1.2.0",
        "features": ["tools", "resources"],
    }
    
# Example specifying metadata
@mcp.resource(
    uri="data://app-status",      # Explicit URI (required)
    name="ApplicationStatus",     # Custom name
    description="Provides the current status of the application.", # Custom description
    mime_type="application/json", # Explicit MIME type
    tags={"monitoring", "status"} # Categorization tags
)

@mcp.resource("resource://system-status")
async def get_system_status(ctx: Context) -> dict:
    """Provides system status information."""
    return {
        "status": "operational",
        "request_id": ctx.request_id
    }

@mcp.resource("resource://{name}/details")
async def get_details(name: str, ctx: Context) -> dict:
    """Get details for a specific name."""
    return {
        "name": name,
        "accessed_at": ctx.request_id
    }

def get_application_status() -> dict:
    """Internal function description (ignored if description is provided above)."""
    return {"status": "ok", "uptime": 12345, "version": mcp.settings.version} # Example usage

@mcp.prompt()
def summarize_request(text: str) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text:\n\n{text}"


if __name__ == "__main__":
    mcp.run(transport='sse', host="127.0.0.1", port=8001)