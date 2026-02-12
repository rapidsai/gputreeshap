#!/bin/bash
# Copyright (c) 2023-2026, NVIDIA CORPORATION.
###############################
# gputreeshap Version Updater #
###############################

## Usage
# bash update-version.sh <new_version>
#
# Where <new_version> is a RAPIDS version, like '26.06.00'.

set -e -u -o pipefail

# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case $arg in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "$VERSION_ARG" ]]; then
                VERSION_ARG="$arg"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="$VERSION_ARG"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "$CLI_RUN_CONTEXT" ]]; then
    RUN_CONTEXT="$CLI_RUN_CONTEXT"
    echo "Using run-context from CLI: $RUN_CONTEXT"
elif [[ -n "${RAPIDS_RUN_CONTEXT:-}" ]]; then
    RUN_CONTEXT="$RAPIDS_RUN_CONTEXT"
    echo "Using run-context from environment: $RUN_CONTEXT"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: $RUN_CONTEXT"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "$NEXT_FULL_TAG" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] $0 <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update $CURRENT_TAG => $NEXT_FULL_TAG (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Update RAPIDS_BRANCH
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

# Centralized version file update
echo "${NEXT_FULL_TAG}" > RAPIDS_VERSION

# CMakeLists
sed_runner 's/'"GPUTreeShap VERSION .* LANGUAGES"'/'"GPUTreeShap VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' CMakeLists.txt

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
done
