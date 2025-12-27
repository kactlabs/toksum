"""
Command-line interface for toksum.

This module provides a comprehensive command-line interface for the toksum library,
allowing users to count tokens and estimate costs for various LLM models directly
from the terminal.

The CLI supports:
    - Token counting for text input or files
    - Cost estimation with detailed breakdowns
    - Listing all supported models by provider
    - Verbose output with detailed information
    - Support for both input and output token pricing

Examples:
    Basic token counting:
    
    .. code-block:: bash
    
        toksum "Hello, world!" gpt-4
        toksum --file input.txt claude-3-opus-20240229
    
    Cost estimation:
    
    .. code-block:: bash
    
        toksum --cost "Your text here" gpt-4
        toksum --cost --output-tokens "Response text" gpt-4
    
    List supported models:
    
    .. code-block:: bash
    
        toksum --list-models
    
    Verbose output:
    
    .. code-block:: bash
    
        toksum --verbose --cost --file large_document.txt gpt-4

Functions:
    main: Main CLI entry point that handles argument parsing and execution
    list_models: Display all supported models organized by provider

The CLI provides comprehensive error handling and user-friendly output formatting
for both simple token counting and detailed cost analysis workflows.
"""

import argparse
import sys
from typing import List, Dict, Any

from .core import TokenCounter, count_tokens, get_supported_models, estimate_cost
from .exceptions import UnsupportedModelError, TokenizationError


def main() -> None:
    """
    Main CLI entry point.
    
    Parses command-line arguments and executes the appropriate toksum functionality.
    Supports token counting, cost estimation, model listing, and file input processing.
    
    The function handles:
        - Argument parsing and validation
        - Text input from command line or file
        - Token counting for specified models
        - Cost estimation with input/output token differentiation
        - Model listing with provider organization
        - Comprehensive error handling and user feedback
        - Verbose output formatting
    
    Command-line Arguments:
        text (str, optional): Text to count tokens for
        model (str, optional): Model name (required unless using --list-models)
        --file, -f (str): Read text from file instead of command line
        --list-models, -l: List all supported models by provider
        --cost, -c: Show cost estimation along with token count
        --output-tokens: Calculate cost for output tokens instead of input
        --verbose, -v: Show detailed output with additional information
    
    Exit Codes:
        0: Success
        1: Error (unsupported model, file not found, tokenization failure, etc.)
    
    Raises:
        SystemExit: On error conditions or user interruption
    
    Examples:
        Basic usage:
        
        .. code-block:: bash
        
            toksum "Hello, world!" gpt-4
            toksum --file document.txt claude-3-opus-20240229
        
        With cost estimation:
        
        .. code-block:: bash
        
            toksum --cost --verbose "Long text content" gpt-4
            toksum --cost --output-tokens "Response text" gpt-4
        
        List models:
        
        .. code-block:: bash
        
            toksum --list-models
    """
    parser = argparse.ArgumentParser(
        description="Count tokens for various LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  toksum "Hello, world!" gpt-4
  toksum --file input.txt claude-3-opus-20240229
  toksum --list-models
  toksum --cost "Your text here" gpt-4
        """
    )
    
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to count tokens for (use --file to read from file)"
    )
    
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name (e.g., gpt-4, claude-3-opus-20240229)"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Read text from file instead of command line argument"
    )
    
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List all supported models"
    )
    
    parser.add_argument(
        "--cost", "-c",
        action="store_true",
        help="Show cost estimation along with token count"
    )
    
    parser.add_argument(
        "--output-tokens",
        action="store_true",
        help="Calculate cost for output tokens instead of input tokens"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        if args.list_models:
            list_models()
            return
        
        if not args.model:
            parser.error("Model name is required unless using --list-models")
        
        # Get text input
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = f.read()
                if args.verbose:
                    print(f"Read {len(text)} characters from {args.file}")
            except FileNotFoundError:
                print(f"Error: File '{args.file}' not found", file=sys.stderr)
                sys.exit(1)
            except Exception as e:
                print(f"Error reading file: {e}", file=sys.stderr)
                sys.exit(1)
        elif args.text:
            text = args.text
        else:
            parser.error("Either provide text as argument or use --file option")
        
        # Count tokens
        try:
            token_count = count_tokens(text, args.model)
            
            if args.verbose:
                print(f"Model: {args.model}")
                print(f"Text length: {len(text)} characters")
                print(f"Token count: {token_count}")
            else:
                print(token_count)
            
            # Show cost estimation if requested
            if args.cost:
                input_cost = estimate_cost(token_count, args.model, input_tokens=True)
                output_cost = estimate_cost(token_count, args.model, input_tokens=False)
                
                if input_cost > 0 or output_cost > 0:
                    if args.verbose:
                        print(f"Estimated input cost: ${input_cost:.6f}")
                        print(f"Estimated output cost: ${output_cost:.6f}")
                    else:
                        cost = output_cost if args.output_tokens else input_cost
                        print(f"${cost:.6f}")
                else:
                    if args.verbose:
                        print("Cost estimation not available for this model")
        
        except UnsupportedModelError as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                print("\nUse --list-models to see supported models", file=sys.stderr)
            sys.exit(1)
        
        except TokenizationError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def list_models() -> None:
    """
    List all supported models organized by provider.
    
    Displays a comprehensive list of all supported models grouped by their
    respective providers (OpenAI, Anthropic, Google, Meta, etc.). The output
    includes model counts per provider and a total count across all providers.
    
    The function:
        - Retrieves all supported models using get_supported_models()
        - Groups models by provider with clear section headers
        - Sorts models alphabetically within each provider
        - Shows model counts for each provider and overall total
        - Formats output for easy readability
    
    Output Format:
        .. code-block:: text
        
            Supported models:
            ==================================================
            
            OPENAI (25 models):
            ------------------------------
              gpt-3.5-turbo
              gpt-4
              gpt-4o
              ...
            
            ANTHROPIC (12 models):
            ------------------------------
              claude-3-haiku-20240307
              claude-3-opus-20240229
              ...
            
            Total: 200+ models
    
    Note:
        This function is typically called when the --list-models CLI flag is used.
        It provides users with a complete overview of available models for token
        counting and cost estimation.
    """
    models = get_supported_models()
    
    print("Supported models:")
    print("=" * 50)
    
    for provider, model_list in models.items():
        print(f"\n{provider.upper()} ({len(model_list)} models):")
        print("-" * 30)
        
        for model in sorted(model_list):
            print(f"  {model}")
    
    print(f"\nTotal: {sum(len(models) for models in models.values())} models")


if __name__ == "__main__":
    main()
