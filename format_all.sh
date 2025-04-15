#!/bin/bash

echo "🧼 formatting all Python files with black..."
pip install black --quiet
black . --line-length 120

echo "✅ done!"
