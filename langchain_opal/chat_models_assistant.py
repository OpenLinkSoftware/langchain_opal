"""ChatOpalAssistant chat model."""


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

class ChatOpalAssistant(BaseChatModel):
    """ChatOpalAssistant chat model.

    Example:
        .. code-block:: python

            import os
            os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

            from langchain_opal import ChatOpalAssistant

            model = ChatOpalAssistant()

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            model.invoke(messages)

    """

    model_name: Optional[str] = None
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

    assistant_id: Optional[str] = "asst_IcfB5bT1ep4SQW5vbNFChnX4"
    """The assistant_id to use."""

    funcs_list: Optional[List[str]] = None
    """Functions list"""

    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    """The timeout for making http request to OPAL API server"""

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

    _thread_id: str = None
    continue_thread: bool = False


    @staticmethod
    def get_assistants_list(
        api_base: Optional[str] = "https://linkeddata.uriburner.com",
        ) -> []:
        openlink_api_key = os.environ["OPENLINK_API_KEY"]
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {openlink_api_key}",
        }

        _url = f"{api_base}/chat/api/assistants"
        with httpx.Client(timeout=Timeout(DEFAULT_REQUEST_TIMEOUT)) as client:
            response = client.get(
                url=_url,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Opal."""
        payload = {
            "assistant_id": self.assistant_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.funcs_list is not None and len(self.funcs_list)>0:
            payload["functions"] = self.funcs_list

        if self.model_name is not None and len(self.model_name)>0:
            payload["model"] = self.model_name
        return payload

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

        payload = self._default_params
        payload["apiKey"] = self.openai_api_key.get_secret_value()

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.openlink_api_key.get_secret_value()}",
        }
        api_run_url = f"{self.api_base}/chat/api/runAssistant"
        api_threads_url = f"{self.api_base}/chat/api/threads"
        thread_id = self._thread_id if self.continue_thread else None

        if thread_id is None:
            _create_url = f"{api_threads_url}?apiKey={self.openai_api_key.get_secret_value()}"
            with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
                response = client.post(
                    url=_create_url,
                    json={},
                    headers=headers,
                )
                response.raise_for_status()
                thread_id = response.text
                if thread_id is None:
                    raise (ValueError("Could not create Thread"))
                if self.continue_thread:
                    self._thread_id = thread_id

        payload["thread_id"] = thread_id
        payload["prompt"] = self._convert_messages_to_prompt(messages)

        # logger.info(f"Sending request to {api_run_url} with payload: {payload} ")
        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=api_run_url,
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
            # logger.info(f"Received response: {message}")
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
        api_run_url = f"{self.api_base}/chat/api/runAssistant"
        api_threads_url = f"{self.api_base}/chat/api/threads"
        thread_id = self._thread_id if self.continue_thread else None

        if thread_id is None:
            _create_url = f"{api_threads_url}?apiKey={self.openai_api_key.get_secret_value()}"
            async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
                response = await client.post(
                    url=_create_url,
                    json={},
                    headers=headers,
                )
                response.raise_for_status()
                thread_id = response.text
                if thread_id is None:
                    raise (ValueError("Could not create Thread"))
                if self.continue_thread:
                    self._thread_id = thread_id

        payload["thread_id"] = thread_id
        payload["prompt"] = self._convert_messages_to_prompt(messages)

        # logger.info(f"Sending request to {api_run_url} with payload: {payload} ")
        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=api_run_url,
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
            # logger.info(f"Received response: {message}")
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

