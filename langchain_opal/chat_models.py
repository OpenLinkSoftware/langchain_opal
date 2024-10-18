from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from httpx import Timeout

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
)

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_core.runnables import run_in_executor

from pydantic import Field, PrivateAttr, SecretStr, model_validator
from typing_extensions import Self


DEFAULT_REQUEST_TIMEOUT = 60.0

class ChatOpal(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.

    Example:

        .. code-block:: python

            model = CustomChatModel(n=2)
            result = model.invoke([HumanMessage(content="hello")])
            result = model.batch([[HumanMessage(content="hello")],
                                [HumanMessage(content="world")]])
    """

    model_name: str = "gpt-4o"
    """Model name to use."""

    temperature: Optional[float] = 0.2
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.2)"""

    stop: Optional[List[str]] = None
    """Sets the stop tokens to use."""

    top_p: Optional[float] = 0.5
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.5)"""

    api_base: Optional[str] = "https://linkeddata.uriburner.com"
    """The base URL for OPAL API."""

    finetune: Optional[str] = "system-data-twingler-config"
    """Finetune mode"""

    funcs_list: Optional[List[str]] = ["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"]
    """Finetune mode"""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """The timeout for making http request to llamafile API server"""

    openlink_api_key: SecretStr = Field(
        alias="openlink_api_key",
        default_factory=secret_from_env(
            "OPENLINK_API_KEY",
            error_message=(
                "You must specify an openlink api key. "
                "You can pass it an argument as `openlink_api_key=...` or "
                "set the environment variable `OPENLINK_API_KEY`."
            ),
        ),
    )
    """OpenLink API key.

    Automatically read from env variable `OPENLINK_API_KEY` if not provided.
    """

    openai_api_key: SecretStr = Field(
        alias="openai_api_key",
        default_factory=secret_from_env(
            "OPENAI_API_KEY",
            error_message=(
                "You must specify an openai api key. "
                "You can pass it an argument as `openai_api_key=...` or "
                "set the environment variable `OPENAI_API_KEY`."
            ),
        ),
    )
    """OpenAI API key.

    Automatically read from env variable `OPENAI_API_KEY` if not provided.
    """

    _chat_id: str = None
    continue_chat: bool = False

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Opal."""
        return {
            "model": self.model_name,
            "type": "user",
            "temperature": self.temperature,
            "top_p": self.top_p,
            "call": self.funcs_list,
            "fine_tune": self.finetune,
        }

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        prompt = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                prompt += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"AI: {message.content}\n"
            elif isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            elif isinstance(message, ToolMessage):
                prompt += f"Tool ({message.tool_name}): {message.content}\n"
            else:
                prompt += f"{message.content}\n"  # For other message types
        return prompt


    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                If generation stops due to a stop token, the stop token itself
                SHOULD BE INCLUDED as part of the output. This is not enforced
                across models right now, but it's a good practice to follow since
                it makes it much easier to parse the output of the model
                downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        # # IBM Watson
        # if self.streaming:
        #     stream_iter = self._stream(
        #         messages, stop=stop, run_manager=run_manager, **kwargs
        #     )
        #     return generate_from_stream(stream_iter)

        payload = self._default_params
        payload["apiKey"] = self.openai_api_key.get_secret_value()

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.openlink_api_key.get_secret_value()}",
        }
        api_url = f"{self.api_base}/chat/api/chatCompletion"
        chat_id = self._chat_id if self.continue_chat else None

        if chat_id is None:
            payload["chat_id"] = self.finetune
            with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
                response = client.post(
                    url=api_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                raw = response.json()
                chat_id = raw.get("chat_id")
                if chat_id is None:
                    raise (ValueError("Could not create Chat"))
                if self.continue_chat:
                    self._chat_id = chat_id

        payload["chat_id"] = chat_id
        payload["question"] = self._convert_messages_to_prompt(messages)

        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message_data = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message_data))

            message = AIMessage(
                content=message_data,
                additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            )

            metadata = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # "token_usage": response_data.get("token_usage", "unknown"),
                    "status": "success",
                    "raw_response": raw
                }
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation], llm_output=metadata)


    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Override the _generate method to implement the chat model logic.

        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                If generation stops due to a stop token, the stop token itself
                SHOULD BE INCLUDED as part of the output. This is not enforced
                across models right now, but it's a good practice to follow since
                it makes it much easier to parse the output of the model
                downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        payload = self._default_params
        payload["apiKey"] = self.openai_api_key.get_secret_value()

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.openlink_api_key.get_secret_value()}",
        }
        api_url = f"{self.api_base}/chat/api/chatCompletion"
        chat_id = self._chat_id if self.continue_chat else None

        if chat_id is None:
            payload["chat_id"] = self.finetune
            async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
                response = await client.post(
                    url=api_url,
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                raw = response.json()
                chat_id = raw.get("chat_id")
                if chat_id is None:
                    raise (ValueError("Could not create Chat"))
                if self.continue_chat:
                    self._chat_id = chat_id

        payload["chat_id"] = chat_id
        payload["question"] = self._convert_messages_to_prompt(messages)

        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=api_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message_data = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message_data))

            message = AIMessage(
                content=message_data,
                additional_kwargs={},  # Used to add additional payload (e.g., function calling request)
            )

            metadata = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    # "token_usage": response_data.get("token_usage", "unknown"),
                    "status": "success",
                    "raw_response": raw
                }

            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation], llm_output=metadata)


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "ChatOpal"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
            "api_base": self.api_base,
            "finetune": self.finetune,
            "funcs_list": self.funcs_list,
            "continue_chat": self.continue_chat,
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="opal",
            ls_model_name=self.model_name,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

