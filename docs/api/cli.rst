Command Line Interface
======================

The CLI module provides a command-line interface for toksum functionality.

.. automodule:: toksum.cli
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: toksum.cli.main

.. autofunction:: toksum.cli.list_models

Usage Examples
--------------

Basic Token Counting
~~~~~~~~~~~~~~~~~~~~~

Count tokens for a simple text string:

.. code-block:: bash

   toksum "Hello, world!" gpt-4

Count tokens from a file:

.. code-block:: bash

   toksum --file document.txt claude-3-opus-20240229

Cost Estimation
~~~~~~~~~~~~~~~

Get cost estimates along with token counts:

.. code-block:: bash

   toksum --cost "Your text here" gpt-4

Calculate output token costs:

.. code-block:: bash

   toksum --cost --output-tokens "Response text" gpt-4

Model Discovery
~~~~~~~~~~~~~~~

List all supported models:

.. code-block:: bash

   toksum --list-models

Verbose Output
~~~~~~~~~~~~~~

Get detailed information:

.. code-block:: bash

   toksum --verbose --cost --file large_document.txt gpt-4

Command Line Arguments
----------------------

Positional Arguments
~~~~~~~~~~~~~~~~~~~~

* ``text`` - Text to count tokens for (optional if using --file)
* ``model`` - Model name (required unless using --list-models)

Optional Arguments
~~~~~~~~~~~~~~~~~~

* ``--file, -f`` - Read text from file instead of command line
* ``--list-models, -l`` - List all supported models
* ``--cost, -c`` - Show cost estimation along with token count
* ``--output-tokens`` - Calculate cost for output tokens instead of input
* ``--verbose, -v`` - Show verbose output with additional details

Exit Codes
-----------

* ``0`` - Success
* ``1`` - Error (unsupported model, file not found, tokenization failure, etc.)