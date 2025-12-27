Core Module
===========

The core module contains the main functionality for token counting across different LLM providers.

.. automodule:: toksum.core
   :members:
   :undoc-members:
   :show-inheritance:

TokenCounter Class
------------------

.. autoclass:: toksum.core.TokenCounter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Convenience Functions
---------------------

.. autofunction:: toksum.core.count_tokens

.. autofunction:: toksum.core.get_supported_models

.. autofunction:: toksum.core.estimate_cost

Model Dictionaries
-------------------

The core module defines comprehensive model dictionaries for all supported providers:

OpenAI Models
~~~~~~~~~~~~~

.. autodata:: toksum.core.OPENAI_MODELS
   :annotation: = {...}

.. autodata:: toksum.core.OPENAI_LEGACY_MODELS
   :annotation: = {...}

.. autodata:: toksum.core.OPENAI_O1_MODELS
   :annotation: = {...}

Anthropic Models
~~~~~~~~~~~~~~~~

.. autodata:: toksum.core.ANTHROPIC_MODELS
   :annotation: = {...}

.. autodata:: toksum.core.ANTHROPIC_LEGACY_MODELS
   :annotation: = {...}

Google Models
~~~~~~~~~~~~~

.. autodata:: toksum.core.GOOGLE_MODELS
   :annotation: = {...}

Meta Models
~~~~~~~~~~~

.. autodata:: toksum.core.META_MODELS
   :annotation: = {...}

Other Provider Models
~~~~~~~~~~~~~~~~~~~~~

The module includes model dictionaries for 20+ additional providers including:

* Mistral (MISTRAL_MODELS)
* Cohere (COHERE_MODELS)
* xAI (XAI_MODELS)
* Alibaba (ALIBABA_MODELS)
* Baidu (BAIDU_MODELS)
* Huawei (HUAWEI_MODELS)
* DeepSeek (DEEPSEEK_MODELS)
* And many more...

Each dictionary maps model names to their tokenization identifiers for approximation algorithms.