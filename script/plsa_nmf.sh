#!/bin/bash

project_home=$(readlink -f $(dirname $0))/..
datapath=$project_home/data/
binpath=$project_home/bin/

example=${datapath}/plsa_nmf/example


${binpath}/plsa_nmf_train $example

