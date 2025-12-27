toksum Documentation
====================

.. image:: https://img.shields.io/pypi/v/toksum.svg
   :target: https://pypi.org/project/toksum/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/toksum.svg
   :target: https://pypi.org/project/toksum/
   :alt: Python versions

.. image:: https://img.shields.io/github/license/kactlabs/toksum.svg
   :target: https://github.com/kactlabs/toksum/blob/main/LICENSE
   :alt: License

**toksum** is a comprehensive Python library for counting tokens across 200+ Large Language Models from 25+ providers. It provides precise token counting for OpenAI models using tiktoken and intelligent approximations for all other providers.

Features
--------

* **Comprehensive Model Support**: 200+ models from 25+ providers
* **Precise Counting**: Exact token counts for OpenAI models using tiktoken
* **Intelligent Approximations**: Calibrated algorithms for other providers
* **Cost Estimation**: Built-in pricing for cost calculations
* **Chat Format Support**: Token counting for conversation messages
* **Case-Insensitive**: Flexible model name matching
* **CLI Interface**: Command-line tool for quick token counting

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install toksum

For OpenAI model support (recommended):

.. code-block:: bash

   pip install toksum[openai]

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from toksum import count_tokens, TokenCounter
   
   # Quick token counting
   tokens = count_tokens("Hello, world!", "gpt-4")
   print(f"Tokens: {tokens}")
   
   # Using TokenCounter for multiple operations
   counter = TokenCounter("gpt-4")
   tokens = counter.count("Hello, world!")
   
   # Chat message format
   messages = [
       {"role": "user", "content": "Hello!"},
       {"role": "assistant", "content": "Hi there!"}
   ]
   total_tokens = counter.count_messages(messages)

Supported Providers
-------------------

toksum supports models from major providers:

* **OpenAI**: GPT-4, GPT-3.5, GPT-4o, O1 models (25+ models)
* **Anthropic**: Claude 3/3.5 (Opus, Sonnet, Haiku), Claude 2 (12+ models)
* **Google**: Gemini Pro/Flash, Gemini 1.5/2.0, PaLM (10+ models)
* **Meta**: LLaMA 2/3/3.1/3.2/3.3 variants (15+ models)
* **Mistral**: Mistral 7B, Mixtral, Large variants (10+ models)
* **Cohere**: Command, Command-R, Command-R+ (8+ models)
* **xAI**: Grok models (4+ models)
* **Chinese Providers**: Alibaba Qwen, Baidu ERNIE, Huawei PanGu, Tsinghua ChatGLM
* **Code Models**: DeepSeek Coder, Replit Code, BigCode StarCoder
* **Open Source**: EleutherAI, Stability AI, TII Falcon, RWKV
* **Enterprise**: Databricks DBRX, Microsoft Phi, Amazon Titan, IBM Granite

API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/core
   api/cli
   api/exceptions
   examples
   changelog

Core Functions
--------------

.. autofunction:: toksum.count_tokens

.. autofunction:: toksum.get_supported_models

.. autofunction:: toksum.estimate_cost

TokenCounter Class
------------------

.. autoclass:: toksum.TokenCounter
   :members:
   :undoc-members:
   :show-inheritance:

Command Line Interface
----------------------

toksum provides a command-line interface for quick token counting:

.. code-block:: bash

   # Basic usage
   toksum "Hello, world!" gpt-4
   
   # From file
   toksum --file document.txt claude-3-opus
   
   # With cost estimation
   toksum --cost "Your text" gpt-4
   
   # List supported models
   toksum --list-models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`