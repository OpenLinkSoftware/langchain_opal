# langchain_opal
Connects Langchain to AI Agents built and deployed using the OpenLink AI Layer (OPAL), leveraging its loosely coupled Data Spaces (including databases, knowledge graphs, and documents).


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

#### `OpalLLM` parameters 
```python
  OpalLLM(
    model_name="gpt-4o",
    temperature=0.2,
    top_p=0.5,
    api_base="https://linkeddata.uriburner.com",
    finetune="system-data-twingler-config",
    funcs_list=["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"],
    request_timeout=60.0
  )
```
 Default values:
- model_name       Default **str** = `"gpt-4o"`
- temperature      Default **float** = `0.2`
- top_p            Default **float** = `0.5`
- api_base         Default **str** = `"https://linkeddata.uriburner.com"`
- finetune         Default **str** = `"system-data-twingler-config"`
- funcs_list       Default **List[str]** = `["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"]`
- request_timeout  Default **float** = DEFAULT_REQUEST_TIMEOUT=`60.0`


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

#### `OpalAssistantLLM` parameters 
```python
  OpalAssistantLLM(
    model_name=None,
    temperature=0.2,
    top_p=0.5,
    api_base="https://linkeddata.uriburner.com",
    assistant_id="asst_IcfB5bT1ep4SQW5vbNFChnX4",
    funcs_list=None,
    request_timeout=60.0
  )
```
 Default values:
- model_name       Default **str** = `None`
- temperature      Default **float** = `0.2`
- top_p            Default **float** = `0.5`
- api_base         Default **str** = `"https://linkeddata.uriburner.com"`
- funcs_list       Default **List[str]** =  `None`
- assistant_id     Default **str** = `"asst_IcfB5bT1ep4SQW5vbNFChnX4"`
- request_timeout  Default **float** = DEFAULT_REQUEST_TIMEOUT=`60.0`


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

#### `ChatOpal` parameters 
```python
  ChatOpal(
    model_name="gpt-4o",
    temperature=0.2,
    top_p=0.5,
    api_base="https://linkeddata.uriburner.com",
    finetune="system-data-twingler-config",
    funcs_list=["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"],
    request_timeout=60.0
  )
```
 Default values:
- model_name       Default **str** = `"gpt-4o"`
- temperature      Default **float** = `0.2`
- top_p            Default **float** = `0.5`
- api_base         Default **str** = `"https://linkeddata.uriburner.com"`
- finetune         Default **str** = `"system-data-twingler-config"`
- funcs_list       Default **List[str]** = `["UB.DBA.sparqlQuery", "DB.DBA.vos_howto_search", "Demo.demo.execute_sql_query", "DB.DBA.graphqlQuery"]`
- request_timeout  Default **float** = DEFAULT_REQUEST_TIMEOUT=`60.0`




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

#### `ChatOpalAssistant` parameters 
```python
  ChatOpalAssistant(
    model_name=None,
    temperature=0.2,
    top_p=0.5,
    api_base="https://linkeddata.uriburner.com",
    assistant_id="asst_IcfB5bT1ep4SQW5vbNFChnX4",
    funcs_list=None,
    request_timeout=60.0
  )
```
 Default values:
- model_name       Default **str** = `None`
- temperature      Default **float** = `0.2`
- top_p            Default **float** = `0.5`
- api_base         Default **str** = `"https://linkeddata.uriburner.com"`
- funcs_list       Default **List[str]** = `None`
- assistant_id     Default **str** = `"asst_IcfB5bT1ep4SQW5vbNFChnX4"`
- request_timeout  Default **float** = DEFAULT_REQUEST_TIMEOUT=`60.0`



