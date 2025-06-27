#!/bin/bash

# ==============================================================================
#  Python App Service Starter (Bash-Only Edition)
#  - Starts, stops, and monitors the Python service.
#  - Usage:
#      bash start.sh         (to start the service)
#      bash start.sh monitor (to view live logs for the current session)
#      bash start.sh stop    (to stop the service and clean up logs)
# ==============================================================================

# --- Configuration ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# --- Script Logic ---
# Assumes the script is run from the project's root directory
PROJECT_ROOT=$(pwd)
APP_NAME="Python App"
PYTHON_APP_FILE="app.py"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"

PID_FILE="$PROJECT_ROOT/app.pid"
LOG_FILE="$PROJECT_ROOT/app.log"

# --- Stop Functionality ---
if [ "$1" == "stop" ]; then
    echo -e "${YELLOW}🛑 Stopping ${APP_NAME} service...${NC}"
    if [ -f "$PID_FILE" ]; then
        echo "   - Stopping ${APP_NAME} (PID: $(cat $PID_FILE))..."
        kill $(cat $PID_FILE)
        rm "$PID_FILE"
    else
        echo "   - ${APP_NAME} not running via this script (no PID file found)."
    fi

    # Clean up the temporary log file
    echo "   - Cleaning up temporary log file..."
    rm -f "$LOG_FILE"

    echo -e "${GREEN}✅ Service stopped.${NC}"
    exit 0
fi

# --- Monitor Functionality ---
if [ "$1" == "monitor" ]; then
    echo -e "${BLUE}👀 Monitoring ${APP_NAME} logs... (Press Ctrl+C to exit)${NC}"
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}Log stream not found. Start the service first with 'bash start.sh'.${NC}"
        exit 1
    fi

    # Clean up background tail process on script exit (e.g., via Ctrl+C)
    trap 'echo -e "\n${YELLOW}🛑 Stopping log monitor...${NC}"; kill $(jobs -p) 2>/dev/null' EXIT

    # Tail the log file with a purple prefix
    tail -n 100 -f "$LOG_FILE" | sed "s/^/${PURPLE}[APP]${NC} /" &

    # Wait for the background process, allowing the user to view logs
    wait
    exit 0
fi


# --- Header for Starting ---
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      Starting ${APP_NAME} Service     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo "" # Newline for spacing

# --- Pre-run Checks ---
if [ ! -f "$PYTHON_APP_FILE" ]; then
    echo -e "${YELLOW}Error: Main application file not found: ${PYTHON_APP_FILE}"
    echo "Please run this script from the project's root directory."
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo -e "${YELLOW}Error: Python virtual environment not found at: ${VENV_PATH}"
    echo "Please ensure you have created a virtual environment (e.g., python -m venv .venv)."
    exit 1
fi

echo "🚀 Launching service in the background..."

# --- Service Launch ---
echo "   - Activating Python virtual environment..."
source "$VENV_PATH"

echo "   - Launching ${PYTHON_APP_FILE}..."
# The '&' runs the command in the background.
# Output is redirected to a temporary log file. 2>&1 merges stderr into stdout.
# The '-u' flag ensures that Python output is unbuffered and appears in the log immediately.
python -u "$PYTHON_APP_FILE" > "$LOG_FILE" 2>&1 &
# '$!' gets the PID of the last background process. We save it to a file.
echo $! > "$PID_FILE"

# Deactivate the virtual environment for the current shell session
deactivate

# --- Final Instructions ---
echo -e "\n${GREEN}✅ ${APP_NAME} service is launching in the background.${NC}"
echo -e "   - The application will run as a background process."
echo -e "\nTo view live logs for this session, run this command:"
echo -e "${YELLOW}bash start.sh monitor${NC}"
echo -e "\nTo stop the service and clean up logs, run this command:"
echo -e "${YELLOW}bash start.sh stop${NC}\n"