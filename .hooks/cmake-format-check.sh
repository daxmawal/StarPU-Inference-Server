#!/bin/bash

# Find modified CMake files staged for commit
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '(CMakeLists.txt|\\.cmake$)')

HAS_ERRORS=0

for FILE in $CHANGED_FILES; do
    # Generate diff to see if file is properly formatted
    DIFF=$(cmake-format "$FILE" | diff -u "$FILE" -)

    if [ ! -z "$DIFF" ]; then
        echo "✗ $FILE is not properly formatted."
        echo "$DIFF"
        HAS_ERRORS=1
    else
        echo "✓ $FILE is correctly formatted."
    fi
done

if [ "$HAS_ERRORS" -eq 1 ]; then
    echo ""
    echo "X Formatting issues detected. Please run:"
    echo "    cmake-format -i <files>"
    exit 1
fi

exit 0