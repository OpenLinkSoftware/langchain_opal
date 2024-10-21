"""OPAL Assistant large language models."""

import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Union,
)

import os
import httpx
from httpx import Timeout

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    LangSmithParams,
    # generate_from_stream,
)

from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, PrivateAttr, SecretStr, model_validator
from typing_extensions import Self

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler('opal_assistant_llm.log')
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)


DEFAULT_REQUEST_TIMEOUT = 60.0


class OpalAssistantLLM(LLM):
    """OpalAssistantLLM large language models.

    Example:
        .. code-block:: python

            import os
            os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

            from langchain_opal import OpalAssistantLLM

            model = OpalAssistantLLM()
            model.invoke("Come up with 10 names for a song about parrots")
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


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model..
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

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
        payload["prompt"] = prompt

        logger.info(f"Sending request to {api_run_url} with payload: {payload} ")
        with httpx.Client(timeout=Timeout(self.request_timeout)) as client:
            response = client.post(
                url=api_run_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            # if run_manager:
            #     run_manager.on_text(message)
            logger.info(f"Received response: {message}")
            return message

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to pass into the model.

        Returns:
            The string generated by the model..
        """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

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
        payload["prompt"] = prompt

        logger.info(f"Sending request to {api_run_url} with payload: {payload} ")
        async with httpx.AsyncClient(timeout=Timeout(self.request_timeout)) as client:
            response = await client.post(
                url=api_run_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            raw = response.json()
            kind = raw.get("kind")
            message = raw.get("data", "")
            if kind is not None and kind == "error":
                raise (ValueError(message))
            # if run_manager:
            #     run_manager.on_text(message)
            logger.info(f"Received response: {message}")
            return message


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
            "api_base": self.api_base,
            "assistant_id": self.assistant_id,
            "funcs_list": self.funcs_list,
            "continue_thread": self.continue_thread,
        }


    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "OpalAssistantLLM"


    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names for secret IDs.
        """
        return {
            "openlink_api_key": "OPENLINK_API_KEY",
            "openai_api_key": "OPENAI_API_KEY",
        }

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="opal_assistant",
            ls_model_name=self.model_name,
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params


