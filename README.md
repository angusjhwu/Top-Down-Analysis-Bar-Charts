# Top Down Analysis Bar Charts

Author: Angus Wu

Last Updated: Sep 6, 2024

## Objective

The top down methodology for performance analysis is a powerful tool to identify performance bottlenecks.
This script generates bar charts to better visualize the output from the Linux `perf stat -e/-M -d` options.


## Dependencies

This code has only been tested with Python 3.11.5 and Intel Cascade Lake

Libraries used: `Pandas`, `Matplotlib`


## Usage

```
# Run your program with Perf.
# You can find the available -e and -M options by reading the output of `perf list`

perf stat -e <events> -M <metrics> -d <your program command> > <logfile>`


# Edit the options in the main/default function in `./tda_barcharts.py`

python3 <path_to>/tda_barcharts.py
```