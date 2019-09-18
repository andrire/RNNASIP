#!/bin/bash
# Copyright (c) 2019 ETH Zurich, Integrated System Laboratory, Renzo Andri

echo "Clone Basic Tests"
git clone -v git@iis-git.ee.ethz.ch:pulp-tests/rt-tests.git
echo "All tests should work except for tests including hyperbus for pulpissio"

