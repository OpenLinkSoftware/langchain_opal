# langchain_opal
OpenLink AI Layer (OPAL) Data Connectivity Middleware Module & Persistence Layer for Langchain


## Installation

```bash
pip install -U git+https://github.com/OpenLinkSoftware/langchain_opal.git

```

- The default OPAL and OPALAssistant server is https://linkeddata.uriburner.com
- The UI for OPAL model  https://linkeddata.uriburner.com/chat/
- The UI for OPALAssistant model  https://linkeddata.uriburner.com/assist/


## How to get OPENLINK_API_KEY
Visit the API Key Generation Page at: https://linkeddata.uriburner.com/oauth/applications.vsp

## LLMS models

### OpalLLM
`OpalLLM` class exposes LLMs from OPAL.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxx"
os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxx"

from langchain_opal import OpalLLM

llm = OpalLLM()
llm.invoke("The meaning of life is")
```

### OpalAssistantLLM
`OpalAssistantLLM` class exposes LLMs from OPALAssistant.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxx"
os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxx"

from langchain_opal import OpalAssistantLLM

llm = OpalAssistantLLM()
llm.invoke("The meaning of life is")
```

## Chat Models
### ChatOpal
`ChatOpal` class exposes chat model from OPAL.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxx"
os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxx"

from langchain_opal import ChatOpal

llm = ChatOpal()
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)

```

### ChatOpalAssistant
`ChatOpalAssistant` class exposes chat model from OPALAssistant.

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxx"
os.environ["OPENLINK_API_KEY"] = "xxxxxxxxxxxxxxxxxxx"

from langchain_opal import ChatOpalAssistant

llm = ChatOpalAssistant()
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)
```
