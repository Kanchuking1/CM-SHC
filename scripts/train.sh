#!/usr/bin/env bash

sbatch ./scripts/slurm/cmshc_mirflickr25k_64bit.sbatch
sbatch ./scripts/slurm/cmshc_mirflickr25k_128bit_nocm.sbatch
sbatch ./scripts/slurm/cmshc_mirflickr25k_128bit.sbatch
sbatch ./scripts/slurm/cmshc_mirflickr25k_128bit_csq.sbatch
sbatch ./scripts/slurm/cmshc_mirflickr25k_128bit_cooc.sbatch
