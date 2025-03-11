from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from typing import Dict, Any, List
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient

class RouterAgent:
    def __init__(self, api_key=None):
        # Initialize with API key from config
        self.api_key = api_key
        
        # 创建模型客户端
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4",
            api_key=self.api_key
        )
        
        # Initialize agents with correct parameters
        self.base_knowledge_agent = AssistantAgent(
            name="base_knowledge_agent",
            model_client=self.model_client,
            system_message="You are an expert in general knowledge. Provide accurate and helpful information. When you have fully answered the user's question, end your message with 'TERMINATE'."
        )

        self.client_data_agent = AssistantAgent(
            name="client_data_agent",
            model_client=self.model_client,
            system_message="You are an expert in handling client-specific data and queries. When you have fully answered the user's question, end your message with 'TERMINATE'."
        )
        
        # 定义选择器提示
        self.selector_prompt = """Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
For general knowledge questions about facts, concepts, or information that would be widely known,
select the base_knowledge_agent.

For questions about specific client data, account information, proprietary systems, or personalized services,
select the client_data_agent.

Only select one agent.
"""
        
        # 定义终止条件
        def termination_condition(message):
            return "TERMINATE" in message.get("content", "")
        
        # Create SelectorGroupChat
        self.team = SelectorGroupChat(
            participants=[self.base_knowledge_agent, self.client_data_agent],
            model_client=self.model_client,
            selector_prompt=self.selector_prompt,
            allow_repeated_speaker=True,  # 注意这里是allow_repeated_speaker而不是allow_repeat_speaker
            max_turns=3,  # 注意这里是max_turns而不是max_round
            termination_condition=termination_condition
        )

    async def process_query(self, query: str) -> str:
        # 直接使用team.run方法
        result = await self.team.run(task=query)
        
        # 提取最终响应并移除终止标记
        if isinstance(result, str):
            return result.replace("TERMINATE", "").strip()
        elif isinstance(result, dict) and "content" in result:
            return result["content"].replace("TERMINATE", "").strip()
        else:
            return "No response was generated. Please try again." 