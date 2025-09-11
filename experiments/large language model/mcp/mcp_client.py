import asyncio
import json
import logging
import os
from fastmcp import Client # ✅ 添加这个
from openai import OpenAI # 确保你使用了 openai-compatible 客户端

# Configure basic logging
logging.basicConfig(level=logging.INFO)

class LLMClient:
    """LLM客户端，负责与大语言模型API通信"""

    def __init__(self, model_name: str, url: str, api_key: str) -> None:
        self.model_name: str = model_name
        self.url: str = url
        self.client = OpenAI(api_key=api_key, base_url=url)

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """发送消息给LLM并获取响应"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            return f"Error communicating with LLM: {str(e)}"


class ChatSession:
    """聊天会话，处理用户输入和LLM响应，并与MCP工具交互"""

    def __init__(self, llm_client: LLMClient, mcp_client: Client, ) -> None:
        self.mcp_client: Client = mcp_client
        self.llm_client: LLMClient = llm_client

    async def process_llm_response(self, llm_response: str) -> str:
        """处理LLM响应，解析工具、资源或提示调用并执行"""
        try:
            # 尝试移除可能的markdown格式
            if llm_response.startswith('```json'):
                llm_response = llm_response.strip('```json').strip('```').strip() #
            
            parsed_json = json.loads(llm_response)

            if "tool" in parsed_json and "arguments" in parsed_json:
                tool_name = parsed_json["tool"]
                tool_args = parsed_json["arguments"]
                tools = await self.mcp_client.list_tools()
                if any(tool.name == tool_name for tool in tools):
                    try:
                        result = await self.mcp_client.call_tool(tool_name, tool_args) #
                        return f"Tool execution result: {json.dumps(result, ensure_ascii=False)}"
                    except Exception as e:
                        error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                        logging.error(error_msg)
                        return error_msg
                return f"No server found with tool: {tool_name}"

            elif "resource" in parsed_json: # New: Handling resource calls
                resource_identifier = parsed_json["resource"] # URI template or direct URI
                resource_args = parsed_json.get("arguments", {})
                
                # It's good practice for the LLM to only request resources it knows about.
                # We assume `list_resources()` provides objects with a 'uri' attribute.
                # This check can be enhanced if resource objects have more specific identifiers or parameter schemas.
                available_resources = await self.mcp_client.list_resources()
                if not any(hasattr(r, 'uri') and r.uri == resource_identifier for r in available_resources):
                    # If the resource_identifier isn't directly in the URI list (e.g., LLM composed a final URI)
                    # this check might be too simplistic. For now, we assume LLM uses listed URIs/templates.
                    logging.warning(f"LLM attempted to call resource '{resource_identifier}' which might not match a listed URI template directly.")
                
                try:
                    # ASSUMPTION: fastmcp.Client has a `call_resource` method.
                    # This method would take the resource URI (template) and arguments for path parameters.
                    # If the URI has no parameters, arguments might be empty.
                    # e.g., call_resource("users://{user_id}/profile", {"user_id": 123})
                    # or call_resource("resource://greeting", {})
                    logging.info(f"Attempting to call resource: {resource_identifier} with args: {resource_args}")
                    result = await self.mcp_client.call_resource(resource_identifier, resource_args)
                    return f"Resource '{resource_identifier}' content: {json.dumps(result, ensure_ascii=False)}"
                except AttributeError:
                    # Fallback or error if `call_resource` doesn't exist
                    error_msg = "Client does not support `call_resource`. This functionality needs to be implemented or `fastmcp` version checked."
                    logging.error(error_msg)
                    return error_msg
                except Exception as e:
                    error_msg = f"Error getting resource '{resource_identifier}': {str(e)}"
                    logging.error(error_msg)
                    return error_msg

            elif "prompt" in parsed_json: # New: Handling prompt calls
                prompt_name = parsed_json["prompt"]
                prompt_args = parsed_json.get("arguments", {})
                
                prompts = await self.mcp_client.list_prompts()
                if any(hasattr(p, 'name') and p.name == prompt_name for p in prompts):
                    try:
                        # ASSUMPTION: fastmcp.Client has a `call_prompt` method.
                        # This method would take the prompt name and arguments for the prompt function.
                        logging.info(f"Attempting to call prompt: {prompt_name} with args: {prompt_args}")
                        result = await self.mcp_client.call_prompt(prompt_name, prompt_args)
                        # Prompt output is typically a string.
                        return f"Prompt '{prompt_name}' output: {result}"
                    except AttributeError:
                        error_msg = "Client does not support `call_prompt`. This functionality needs to be implemented or `fastmcp` version checked."
                        logging.error(error_msg)
                        return error_msg
                    except Exception as e:
                        error_msg = f"Error executing prompt '{prompt_name}': {str(e)}"
                        logging.error(error_msg)
                        return error_msg
                return f"No server found with prompt: {prompt_name}"
            
            return llm_response # If not a known action JSON, return as is
        except json.JSONDecodeError:
            # 如果不是JSON格式，直接返回原始响应
            return llm_response #
        except Exception as e:
            # Catch any other unexpected errors during processing
            error_msg = f"Error processing LLM response: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return error_msg


    async def start(self, system_message) -> None:
        """启动聊天会话的主循环"""
        messages = [{"role": "system", "content": system_message}]
        while True:
            try:
                user_input = input("用户: ").strip() # Removed .lower() to preserve case for prompts if needed
                if user_input.lower() in ["quit", "exit", "退出"]:
                    print('AI助手退出') #
                    break
                messages.append({"role": "user", "content": user_input})

                llm_response_text = self.llm_client.get_response(messages)
                print(f"助手 (raw): {llm_response_text}") # Log the raw response for debugging

                # Process LLM response for potential tool/resource/prompt calls
                processed_result = await self.process_llm_response(llm_response_text)

                # If LLM response was an action (tool/resource/prompt call) that got executed
                if processed_result != llm_response_text:
                    print(f"助手 (action result): {processed_result}")
                    messages.append({"role": "assistant", "content": llm_response_text}) # Record LLM's action request
                    messages.append({"role": "system", "content": processed_result})     # Record action's result

                    # Get final natural language response from LLM based on action's result
                    final_llm_response = self.llm_client.get_response(messages)
                    print(f"助手: {final_llm_response}")
                    messages.append({"role": "assistant", "content": final_llm_response})
                else:
                    # LLM response was not an action, or action failed in a way that returned original text
                    print(f"助手: {llm_response_text}")
                    messages.append({"role": "assistant", "content": llm_response_text})

            except KeyboardInterrupt:
                print('AI助手退出') #
                break
            except Exception as e:
                logging.error(f"Error in chat loop: {e}", exc_info=True)
                print("发生了一个错误，请重试。")


async def main():
    # Ensure environment variables are loaded if you use them for API keys, etc.
    # Example: api_key = os.getenv("DEEPSEEK_API_KEY")
    # For this example, using hardcoded values as in the original.
    deepseek_api_key = '' # Replace with your actual key or load from env
    deepseek_url = 'https://api.deepseek.com' #

    async with Client("http://127.0.0.1:8001/sse") as mcp_client: #
        llm_client = LLMClient(
            model_name='deepseek-chat', #
            api_key=deepseek_api_key,
            url=deepseek_url
        )

        tools = await mcp_client.list_tools() #
        # Assuming tool objects have a __dict__ representation or similar serializable format
        tools_description_list = []
        for tool in tools:
            try:
                tools_description_list.append(tool.model_dump()) # pydantic's model_dump if available
            except AttributeError:
                tools_description_list.append(tool.__dict__) # fallback to __dict__
        tools_description = json.dumps(tools_description_list, ensure_ascii=False) #
        
        resources = await mcp_client.list_resources() #
        resources_description_list = []
        for r_obj in resources:
            item_dict = {}
            try:
                item_dict = r_obj.model_dump()
            except AttributeError:
                item_dict = r_obj.__dict__

            # Convert AnyUrl fields to strings
            for key, value in item_dict.items():
                # Check if the value is an instance of pydantic.AnyUrl
                # We need to import AnyUrl or check by type name if direct import is not feasible/desired
                # A common way Pydantic AnyUrl objects behave is that str(value) gives the URL string.
                if hasattr(value, 'scheme') and hasattr(value, 'host') and hasattr(value, 'path'): # Heuristic for Pydantic URL types
                    # More robustly, if you know it's from Pydantic: from pydantic import AnyUrl; isinstance(value, AnyUrl)
                    item_dict[key] = str(value)
                # Also handle if value is a list/dict containing AnyUrl (recursive conversion might be needed for complex cases)
                # For now, assuming AnyUrl is a top-level value in the dumped dict.

            resources_description_list.append(item_dict)
        resources_description = json.dumps(resources_description_list, ensure_ascii=False) #, indent=4)

        
        prompts = await mcp_client.list_prompts() #
        prompts_description_list = []
        for p_obj in prompts:
            try:
                # Similar customization might be needed for prompt objects.
                # Example: {'name': p_obj.name, 'description': p_obj.description, 'parameters': p_obj.parameters}
                prompts_description_list.append(p_obj.model_dump())
            except AttributeError:
                prompts_description_list.append(p_obj.__dict__)

        prompts_description = json.dumps(prompts_description_list, ensure_ascii=False)

        print(f"可用工具: {tools_description}")
        print(f"可用资源: {resources_description}")
        print(f"可用提示: {prompts_description}")

        system_message = f'''
                你是一个智能助手，严格遵循以下协议返回响应：

                可用工具：{tools_description}
                可用资源：{resources_description}
                可用提示：{prompts_description}

                响应规则：
                1. 当需要调用工具时，返回严格符合以下格式的纯净JSON：
                   {{"tool": "tool-name", "arguments": {{"arg-name": "value", ...}}}}
                2. 当需要获取资源时，返回严格符合以下格式的纯净JSON：
                   {{"resource": "resource-uri-template", "arguments": {{"param_name": "value", ...}}}}
                   (使用 '可用资源' 列表中的 'uri' 作为 "resource-uri-template"。 'arguments' 用于填充URI模板中的路径参数，如果URI没有参数则为空对象 {{}}。)
                3. 当需要使用提示模板生成文本时，返回严格符合以下格式的纯净JSON：
                   {{"prompt": "prompt-name", "arguments": {{"arg-for-prompt": "value", ...}}}}
                   (使用 '可用提示' 列表中的 'name' 作为 "prompt-name"。)
                4. 响应的JSON必须是纯净的，不包含任何Markdown标记 (如 ```json) 或自然语言解释。
                5. 数值参数必须是number类型，不要使用字符串表示数字。

                例如 (工具 - multiply):
                用户: 单价88.5买235个多少钱？
                助手: {{"tool":"multiply","arguments":{{"a":88.5,"b":235}}}}

                例如 (资源 - 获取用户资料，假设资源URI模板为 "users://{{user_id}}/profile"):
                用户: 获取用户007的资料。
                助手: {{"resource":"users://{{user_id}}/profile","arguments":{{"user_id":"007"}}}}

                例如 (资源 - 获取静态问候语，假设资源URI为 "resource://greeting"):
                用户: 给我一个问候语。
                助手: {{"resource":"resource://greeting","arguments":{{}}}}

                例如 (提示 - 总结文本，假设提示名称为 "summarize_request"):
                用户: 请总结一下这段话："AI正在改变世界"。
                助手: {{"prompt":"summarize_request","arguments":{{"text":"AI正在改变世界"}}}}

                在收到工具、资源或提示的执行结果后：
                - 将结构化数据（如JSON结果）或文本结果转化为自然的、对话式的回应。
                - 保持回复简洁但信息丰富。
                - 聚焦于最相关的信息。
                - 使用用户问题中的适当上下文。
                - 避免简单重复执行结果的原始数据。
                '''
        
        chat_session = ChatSession(llm_client=llm_client, mcp_client=mcp_client) #
        await chat_session.start(system_message=system_message) #

if __name__ == "__main__":
    try:
        asyncio.run(main()) #
    except KeyboardInterrupt:
        print("程序被用户中断。")
    except Exception as e:
        logging.error(f"应用程序主错误: {e}", exc_info=True)
        
    """
    用户: 现在要购买一批货，单价是 1034.32423，数量是 235326。商家后来又说，可以在这个基础上，打95折，折后总价是多少？
    助手:  {
    "tool": "multiply",
    "arguments": {
        "a": 1034.32423,
        "b": 235326
    }
    }
    助手:  {
    "tool": "multiply",
    "arguments": {
        "a": 243403383.74898,
        "b": 0.95
    }
    }
    助手:  折后总价是231233214.56。
    用户: 我和商家关系比较好，商家说，可以在上面的基础上，再返回两个点，最后总价是多少？
    助手:  {
    "tool": "multiply",
    "arguments": {
        "a": 231233214.56153098,
        "b": 0.98
    }
    }
    助手:  最终总价是226608550.27。
    用户: quit
    AI助手退出。
    """