#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#####################
# GPUTreeShap Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH

# Activate common conda env
source activate gdf

# Check for copyright headers in the files modified currently
COPYRIGHT=`python ci/checks/copyright.py 2>&1`
CR_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$CR_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi


# Check for a consistent code format
FORMAT=`cpplint --recursive GPUTreeShap tests example 2>&1`
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
