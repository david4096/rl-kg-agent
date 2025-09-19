"""LangChain Agent Framework - Migrated from dbclshackathon"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

DEFAULT_LANGCHAIN_AGENT_PROMPT = (
    "You are a helpful assistant."
)


class LangChainAgent:
    """Logic for the creation of the Langchain Agent.
    """

    def __init__(
        self,
        tools = [],
        prompt = DEFAULT_LANGCHAIN_AGENT_PROMPT,
        verbose = True,
        stream_handler = None,
        azure_deployment_name = None,
        api_key = None,
        azure_endpoint = None,
        api_version = None,
        temperature = None
    ):
        """Initialise the object's attributes.

        Parameters
        ----------
        tools : list
            list of the tools for the agent to run
        prompt : str
            the question being posed to the LLM by the user
        verbose : bool
            set to true if you want to see all of the LLMs steps
        stream_handler : object
            the stream handler to be used for the LLM
        azure_deployment_name : str
            the deployment name needed to access the Azure
            OpenAI LLM
        api_key : str
            the api key needed to access the Azure OpenAI LLM
        azure_endpoint : str
            the url of the Azure OpenAI LLM
        api_version : str
            the version to access of the Azure OpenAI LLM
        temperature : float
            the tempertaure to set the Azure OpenAI LLM to
        """

        self.tools = tools
        self.prompt = prompt
        self.verbose = verbose
        self.stream_handler = stream_handler
        self.azure_deployment_name = azure_deployment_name
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.temperature = temperature

    def initialise_openai_llm(self):
        """Connects to the OpenAI LLM to be used.

        Returns
        -------
        llm : object
            the connected llm to the Azure OpenAI LLM
        """
        # Extract base endpoint from full URL if needed
        azure_endpoint = self.azure_endpoint
        if azure_endpoint and '/openai/deployments/' in azure_endpoint:
            # Extract base URL (everything before /openai/deployments/)
            azure_endpoint = azure_endpoint.split('/openai/deployments/')[0] + '/'
        
        if self.stream_handler:
            llm = AzureChatOpenAI(
                azure_deployment = self.azure_deployment_name,
                api_key = self.api_key,
                azure_endpoint = azure_endpoint,
                api_version = self.api_version,
                temperature = self.temperature,
                streaming = True,
                callbacks = [self.stream_handler]
            )
        else:
            llm = AzureChatOpenAI(
                azure_deployment = self.azure_deployment_name,
                api_key = self.api_key,
                azure_endpoint = azure_endpoint,
                api_version = self.api_version,
                temperature = self.temperature,
                streaming = True
            )

        return llm

    def make_langchain_agent(self):
        """Create the LangChain Agent needed for running tools.

        Returns
        -------
        agent_executor : object
            the LangChain agent that is to be used for running tools
        """

        # Initialise the LLM
        llm = self.initialise_openai_llm()

        # Convert string prompt to ChatPromptTemplate if needed
        if isinstance(self.prompt, str):
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.prompt),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
        else:
            prompt = self.prompt

        # Create the OpenAI tools agent
        agent = create_openai_tools_agent(llm, self.tools, prompt)

        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )

        return agent_executor