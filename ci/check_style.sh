#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

# Ignore errors
set +e
RETVAL="0"

. /opt/conda/etc/profile.d/conda.sh

# Check for a consistent code format
pip install cpplint
FORMAT=$(cpplint --recursive GPUTreeShap tests example benchmark 2>&1)
FORMAT_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$FORMAT_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: cpplint format check; begin output\n\n"
  echo -e "$FORMAT"
  echo -e "\n\n>>>> FAILED: cpplint format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: cpplint format check\n\n"
fi

exit $RETVAL
