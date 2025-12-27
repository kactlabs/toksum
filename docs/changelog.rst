Changelog
=========

This document tracks all notable changes to the toksum library.

Version 0.7.0 (Current)
------------------------

**Major Documentation Enhancement:**
- ✅ **Comprehensive Sphinx Documentation** - Added professional-grade documentation for all modules
- ✅ **Enhanced Docstrings** - All functions, classes, and methods now have detailed Google/NumPy style docstrings
- ✅ **Rich Examples** - Code examples for every major feature and use case
- ✅ **API Reference** - Complete API documentation with cross-references
- ✅ **Usage Guides** - Comprehensive examples including advanced patterns and integrations
- ✅ **Error Handling** - Detailed exception documentation with handling patterns
- ✅ **Build Tools** - Convenient documentation build and serve scripts
- ✅ **Mobile Responsive** - Read the Docs theme with search functionality

**Documentation Features:**
- 📚 200+ models from 25+ providers fully documented
- 💡 Rich code examples with syntax highlighting
- 🛡️ Comprehensive error handling patterns
- ⚡ Performance optimization guidance
- 🔗 Cross-referenced API documentation
- 📱 Mobile-responsive design
- 🔍 Full-text search functionality

**Build System:**
- Sphinx configuration with autodoc and Napoleon extensions
- Read the Docs theme integration
- Automated API reference generation
- Local development server support

Version 0.6.0 (Previous)
-------------------------

**Features:**
- Comprehensive support for 200+ models from 25+ providers
- Precise token counting for OpenAI models using tiktoken
- Intelligent approximation algorithms for all other providers
- Cost estimation with USD/INR currency support
- Chat message format token counting
- Case-insensitive model name matching
- Command-line interface with comprehensive options

**Supported Providers:**
- OpenAI (GPT-4, GPT-3.5, GPT-4o, O1 models, embeddings)
- Anthropic (Claude 3/3.5, Claude 2, Instant)
- Google (Gemini Pro/Flash, Gemini 1.5/2.0, PaLM)
- Meta (LLaMA 2/3/3.1/3.2/3.3 variants)
- Mistral (Mistral 7B, Mixtral, Large variants)
- Cohere (Command, Command-R, Command-R+)
- xAI (Grok models)
- Alibaba (Qwen models)
- Baidu (ERNIE models)
- Huawei (PanGu models)
- Yandex (YaLM models)
- DeepSeek (Coder, VL, LLM models)
- Tsinghua (ChatGLM models)
- Databricks (DBRX, Dolly models)
- Voyage AI (embedding models)
- And 10+ more providers

**API:**
- ``count_tokens(text, model)`` - Quick token counting
- ``TokenCounter(model)`` - Reusable token counter class
- ``get_supported_models()`` - List all supported models
- ``estimate_cost(tokens, model)`` - Cost estimation

**CLI:**
- Basic token counting: ``toksum "text" model``
- File input: ``toksum --file document.txt model``
- Cost estimation: ``toksum --cost "text" model``
- Model listing: ``toksum --list-models``

**Error Handling:**
- ``UnsupportedModelError`` - For unsupported models
- ``TokenizationError`` - For tokenization failures
- ``EmptyTextError`` - For empty text validation
- ``InvalidTokenError`` - For token-specific issues

Future Versions
---------------

**Planned Features:**
- Additional model providers as they become available
- Enhanced approximation algorithms based on user feedback
- Streaming token counting for large texts
- Batch processing optimizations
- Integration with more LLM libraries

**Version History:**
- v0.6.0 - Current comprehensive release
- v0.5.x - Previous versions with limited provider support
- v0.4.x - Early releases with basic OpenAI support
- v0.3.x - Initial public releases
- v0.2.x - Beta versions
- v0.1.x - Alpha versions

Contributing
------------

We welcome contributions to expand model support and improve approximation accuracy. Please see the project repository for contribution guidelines.

**Areas for Contribution:**
- New model provider support
- Approximation algorithm improvements
- Documentation enhancements
- Test coverage expansion
- Performance optimizations

**Reporting Issues:**
- Model support requests
- Accuracy improvements
- Bug reports
- Feature requests

License
-------

toksum is released under the MIT License. See the LICENSE file for details.