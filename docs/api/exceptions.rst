Exceptions
==========

The exceptions module defines custom exception classes for toksum error handling.

.. automodule:: toksum.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Exception Hierarchy
--------------------

.. code-block:: text

   ToksumError (base)
   ├── UnsupportedModelError
   ├── ModelNotFoundError  
   ├── TokenizationError
   │   ├── InvalidTokenError
   │   └── EmptyTextError

Base Exception
--------------

.. autoexception:: toksum.exceptions.ToksumError
   :members:
   :show-inheritance:

Model-Related Exceptions
------------------------

.. autoexception:: toksum.exceptions.UnsupportedModelError
   :members:
   :show-inheritance:

.. autoexception:: toksum.exceptions.ModelNotFoundError
   :members:
   :show-inheritance:

Tokenization Exceptions
-----------------------

.. autoexception:: toksum.exceptions.TokenizationError
   :members:
   :show-inheritance:

.. autoexception:: toksum.exceptions.InvalidTokenError
   :members:
   :show-inheritance:

.. autoexception:: toksum.exceptions.EmptyTextError
   :members:
   :show-inheritance:

Error Handling Examples
-----------------------

Basic Error Handling
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from toksum import TokenCounter
   from toksum.exceptions import UnsupportedModelError, TokenizationError
   
   try:
       counter = TokenCounter("unknown-model")
       tokens = counter.count("Hello, world!")
   except UnsupportedModelError as e:
       print(f"Model not supported: {e.model}")
       print(f"Available models: {e.supported_models}")
   except TokenizationError as e:
       print(f"Tokenization failed: {e}")

Comprehensive Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from toksum import count_tokens
   from toksum.exceptions import ToksumError
   
   try:
       tokens = count_tokens("Hello!", "gpt-4")
   except ToksumError as e:
       print(f"Toksum error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Error Context Information
-------------------------

All toksum exceptions provide rich context information:

* **Model information**: Which model caused the error
* **Text previews**: Sample of problematic text (truncated for privacy)
* **Supported alternatives**: List of valid options when applicable
* **Detailed messages**: Clear descriptions of what went wrong