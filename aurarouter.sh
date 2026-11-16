#!/usr/bin/env sh

FOUND_PYTHON=""

test_valid_python() {
    cmd="$1"
    if [ -x "$cmd" ]; then
        if "$cmd" -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

if command -v python3 >/dev/null 2>&1 && test_valid_python "$(command -v python3)"; then
    FOUND_PYTHON=$(command -v python3)
elif command -v python >/dev/null 2>&1 && test_valid_python "$(command -v python)"; then
    FOUND_PYTHON=$(command -v python)
else
    for p in /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3 /opt/homebrew/bin/python /usr/local/bin/python; do
        if test_valid_python "$p"; then
            FOUND_PYTHON="$p"
            break
        fi
    done
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -z "$FOUND_PYTHON" ]; then
    if test_valid_python "$SCRIPT_DIR/../auraseed/.bootstrapped_python/bin/python"; then
        FOUND_PYTHON="$SCRIPT_DIR/../auraseed/.bootstrapped_python/bin/python"
    elif test_valid_python "$HOME/.local/share/auraseed/base_python/bin/python"; then
        FOUND_PYTHON="$HOME/.local/share/auraseed/base_python/bin/python"
    elif test_valid_python "$HOME/Library/Application Support/auraseed/base_python/bin/python"; then
        FOUND_PYTHON="$HOME/Library/Application Support/auraseed/base_python/bin/python"
    fi
fi

if [ -z "$FOUND_PYTHON" ]; then
    echo "[AuraRouter Bootstrapper] Error: Could not locate a base Python installation on host." >&2
    exit 1
fi

PARENT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOPATH="$SCRIPT_DIR/src"

for DEP in auragrid-py auragrid-api benchwarmer; do
    if [ -d "$PARENT_DIR/$DEP/src" ]; then
        AUTOPATH="$AUTOPATH:$PARENT_DIR/$DEP/src"
    elif [ -d "$PARENT_DIR/$DEP" ]; then
        AUTOPATH="$AUTOPATH:$PARENT_DIR/$DEP"
    fi
done

if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH="$AUTOPATH:$PYTHONPATH"
else
    export PYTHONPATH="$AUTOPATH"
fi

exec "$FOUND_PYTHON" -m aurarouter "$@"

