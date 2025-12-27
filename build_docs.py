#!/usr/bin/env python3
"""
Build and optionally serve toksum documentation.

This script provides a convenient way to build the Sphinx documentation
and optionally serve it locally for viewing.
"""

import os
import sys
import subprocess
import webbrowser
import http.server
import socketserver
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error output: {e.stderr}")
        return False


def build_docs():
    """Build the Sphinx documentation."""
    print("Building Sphinx documentation...")
    docs_dir = Path(__file__).parent / "docs"
    
    if not docs_dir.exists():
        print("Error: docs directory not found!")
        return False
    
    # Build HTML documentation
    success = run_command("make html", cwd=docs_dir)
    
    if success:
        html_dir = docs_dir / "_build" / "html"
        index_file = html_dir / "index.html"
        
        if index_file.exists():
            print(f"✓ Documentation built successfully!")
            print(f"  HTML files: {html_dir}")
            print(f"  Main page: {index_file}")
            return True
        else:
            print("Error: HTML files not generated!")
            return False
    else:
        print("Error: Failed to build documentation!")
        return False


def serve_docs(port=8000):
    """Serve the documentation locally."""
    docs_dir = Path(__file__).parent / "docs" / "_build" / "html"
    
    if not docs_dir.exists():
        print("Error: Built documentation not found. Run build first.")
        return False
    
    print(f"Serving documentation at http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        os.chdir(docs_dir)
        with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
            # Open browser
            webbrowser.open(f"http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return True
    except OSError as e:
        print(f"Error starting server: {e}")
        return False


def clean_docs():
    """Clean the documentation build directory."""
    print("Cleaning documentation build files...")
    docs_dir = Path(__file__).parent / "docs"
    
    success = run_command("make clean", cwd=docs_dir)
    if success:
        print("✓ Documentation build files cleaned!")
    else:
        print("Error: Failed to clean build files!")
    
    return success


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build and serve toksum documentation")
    parser.add_argument("action", choices=["build", "serve", "clean", "build-serve"], 
                       help="Action to perform")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port for serving documentation (default: 8000)")
    
    args = parser.parse_args()
    
    if args.action == "build":
        success = build_docs()
        sys.exit(0 if success else 1)
    
    elif args.action == "serve":
        success = serve_docs(args.port)
        sys.exit(0 if success else 1)
    
    elif args.action == "clean":
        success = clean_docs()
        sys.exit(0 if success else 1)
    
    elif args.action == "build-serve":
        if build_docs():
            serve_docs(args.port)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()