"""
Command-line interface for toksum: Count tokens and estimate cost for various LLMs.
"""

import argparse
import sys
import json
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any

from .core import TokenCounter, estimate_cost
from .exceptions import UnsupportedModelError, TokenizationError
from .model_registry import get_supported_models, is_supported_model


def validate_text(text: Optional[str]) -> str:
    """
    Validate input text.

    Args:
        text (Optional[str]): The input text.

    Returns:
        str: Validated text.

    Raises:
        ValueError: If the text is empty or None.
    """
    if not text or not text.strip():
        raise ValueError("Input text is empty.")
    return text


def count_tokens(text: str, model: str) -> int:
    """
    Count tokens in text for a specific model.
    
    Args:
        text: Input text
        model: Model name
        
    Returns:
        Number of tokens
    """
    counter = TokenCounter(model)
    return counter.count(text)


def process_batch(
    directory: str,
    model: str,
    verbose: bool = False,
    cost: bool = False,
    output_tokens: bool = False,
    output_format: str = "text"
) -> None:
    """
    Process all `.txt` files in a directory and output token statistics.

    Args:
        directory (str): Directory path.
        model (str): Model name.
        verbose (bool): Show detailed output.
        cost (bool): Include cost estimation.
        output_tokens (bool): Estimate output token cost instead of input.
        output_format (str): One of 'text', 'json', or 'csv'.
    """
    # REMOVED DUPLICATE IMPORT: from .core import count_tokens, estimate_cost
    
    results = []
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        print(f"Error: Directory '{directory}' not found", file=sys.stderr)
        sys.exit(1)

    txt_files = list(directory_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{directory}'", file=sys.stderr)
        sys.exit(1)

    for file_path in txt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = validate_text(f.read())

            token_count = count_tokens(text, model)
            result = {
                "file": str(file_path),
                "characters": len(text),
                "tokens": token_count
            }

            if cost:
                price = estimate_cost(token_count, model, input_tokens=not output_tokens)
                result["estimated_cost_usd"] = price

            results.append(result)

            if verbose:
                print(f"Processed {file_path.name}: {len(text)} chars, {token_count} tokens")

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    if output_format == "json":
        print(json.dumps(results, indent=2))
    elif output_format == "csv":
        if results:
            writer = csv.DictWriter(sys.stdout, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:  # text
        for res in results:
            print(f"File: {res['file']}")
            print(f"  Characters: {res['characters']}")
            print(f"  Tokens: {res['tokens']}")
            if "estimated_cost_usd" in res:
                print(f"  Cost: ${res['estimated_cost_usd']:.6f}")
            print()


def list_models() -> None:
    """Print all supported models grouped by provider."""
    models = get_supported_models()
    print("Supported models:")
    print("=" * 50)

    for provider, model_list in models.items():
        print(f"\n{provider.upper()} ({len(model_list)} models):")
        print("-" * 30)
        for model in sorted(model_list):
            print(f"  {model}")

    total = sum(len(lst) for lst in models.values())
    print(f"\nTotal: {total} models")


def main() -> None:
    """Main CLI entry point."""
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

    parser.add_argument("text", nargs="?", help="Text to count tokens for (use --file to read from file)")
    parser.add_argument("model", nargs="?", help="Model name (e.g., gpt-4, claude-3-opus-20240229)")

    parser.add_argument("--file", "-f", help="Read text from a file")
    parser.add_argument("--batch-dir", "-b", help="Directory to process all .txt files in batch mode")
    parser.add_argument("--output-format", choices=["text", "json", "csv"], default="text", help="Output format")

    parser.add_argument("--list-models", "-l", action="store_true", help="List all supported models")
    parser.add_argument("--cost", "-c", action="store_true", help="Show cost estimation with token count")
    parser.add_argument("--output-tokens", action="store_true", help="Calculate cost for output tokens")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    try:
        if args.list_models:
            list_models()
            return

        if args.batch_dir:
            if not args.model:
                parser.error("Model is required for batch processing")
            process_batch(
                args.batch_dir,
                args.model,
                verbose=args.verbose,
                cost=args.cost,
                output_tokens=args.output_tokens,
                output_format=args.output_format
            )
            return

        if not args.model:
            parser.error("Model name is required unless using --list-models or --batch-dir")

        # Read input text
        if args.file:
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    text = validate_text(f.read())
                    if args.verbose:
                        print(f"Read {len(text)} characters from {args.file}")
            except FileNotFoundError:
                print(f"Error: File '{args.file}' not found", file=sys.stderr)
                sys.exit(1)
        elif args.text:
            text = validate_text(args.text)
        else:
            parser.error("Either provide text as argument, use --file, or --batch-dir")

        # Token counting and cost
        token_count = count_tokens(text, args.model)

        if args.verbose:
            print(f"Model: {args.model}")
            print(f"Text length: {len(text)} characters")
            print(f"Token count: {token_count}")
        else:
            print(token_count)

        if args.cost:
            price = estimate_cost(token_count, args.model, input_tokens=not args.output_tokens)
            if args.verbose:
                token_type = "output" if args.output_tokens else "input"
                print(f"Estimated {token_type} cost: ${price:.6f}")
            else:
                print(f"${price:.6f}")

    except UnsupportedModelError as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            print("\nUse --list-models to see supported models", file=sys.stderr)
        sys.exit(1)
    except TokenizationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
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


if __name__ == "__main__":
    main()
