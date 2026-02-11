#!/bin/bash
# Copyright (c) 2023-2026, NVIDIA CORPORATION.
###############################
# gputreeshap Version Updater #
###############################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG="${NEXT_MAJOR}.${NEXT_MINOR}"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}.bak"
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# CMakeLists
sed_runner 's/'"GPUTreeShap VERSION .* LANGUAGES"'/'"GPUTreeShap VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' CMakeLists.txt

# Update project_number (RAPIDS_VERSION) in the CPP doxygen file
sed_runner "s/\(PROJECT_NUMBER.*=\).*/\1 \"${NEXT_SHORT_TAG}\"/g" Doxyfile.in

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
