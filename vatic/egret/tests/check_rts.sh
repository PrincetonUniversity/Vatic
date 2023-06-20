#!/bin/bash

vatic-det RTS-GMLC 2020-02-11 2 --solver cbc --threads 1 \
            --ruc-mipgap=0.02 --reserve-factor=0.05 --output-detail 2 --sced-horizon=4 \
            --lmps --lmp-shortfall-costs

# python vatic/tests/compare_outputs.py output.p.gz vatic/tests/resources/output_2020-02-11.p.gz
