"""
* TODO: thing in argparse that lets specify valid options
"""

import argparse
import sys
from pathlib import Path
import shlex
from pprint import pprint
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gics.help import parse_edi, AttrDict
import csv

pd.options.display.max_rows = 1000

init_fn = Path(r'G:\python_projs\gics\.der_e')      # TODO: option to use different file
with init_fn.open('r') as fp:
    init_args = fp.read()

parser = argparse.ArgumentParser(description='Derive electric fields')
parser.add_argument('-f', action='append', dest='fns', type=str, default=[], help='time series file to include in processing')
parser.add_argument('-a', action="store", dest="hs_alpha", type=float, help='half-space alpha value')      # TODO: can put ranges on this e.g. choices=[Range(0.0, 1.0)]. see https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
parser.add_argument('-i', action="store", dest="it_fn", type=Path, help='impedance tensor filename')      # TODO: can put ranges on this e.g. choices=[Range(0.0, 1.0)]. see https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
parser.add_argument('-o', action="store", dest="out_fn", type=Path, help='output filename (if not specified will be outputted to stdout)')
parser.add_argument('--max-ts-gap', action="store", dest="max_ts_gap", type=int, help='maximum number of missing samples that will interpolate')
parser.add_argument('--ts-gap-interp-method', action="store", dest="ts_gap_interp_method", type=str, help='interpolation method to use for time series gaps')       # TODO: these can't be optional
parser.add_argument('--lb', action="store", dest="lb", type=float, help='lower bound for bandpass filter (log scale)')       # TODO: these can't be optional
parser.add_argument('--ub', action="store", dest="ub", type=float, help='upper bound for bandpass filter (log scale)')       # TODO: these can't be optional
# TODO: zero pad to a certain number of days? or a certain number of samples?
parser.add_argument('-c', action="store", dest="col_order", type=str, help='order of columns')
parser.add_argument('--figs', action="store_true", dest="show_figs", help='show figures')

init_args = shlex.split(init_args, comments=True)
all_args = init_args+sys.argv[1:]

args = parser.parse_args(all_args)

temp = []
for fn in args.fns:
    if '*' in fn:
        star_idx = fn.index('*')
        path = Path(fn[:star_idx])
        glob_pat = fn[star_idx:]
        for glob_fn in path.glob(glob_pat):
            temp.append(glob_fn)
    else:
        single_file = Path(fn).absolute()
        if not single_file.is_file():
            sys.exit('{} does not exist'.format(single_file))
        temp.append(single_file)
args.fns = temp

# ======================================================================================================================
# parse impedance tensor
Z, (Z_ll, Z_ul) = parse_edi(args.it_fn)

# fix col order
if args.col_order != None:
    args.col_order = [int(x) for x in args.col_order.split(',')]
else:
    args.col_order = [6,7,11,12]

# ======================================================================================================================
# from https://stackoverflow.com/questions/12269528/using-python-pandas-to-parse-csv-with-date-in-format-year-day-hour-min-sec
def dt_parser(Y, m, d, H, M, S):
    return np.datetime64('{}-{}-{}T{}:{}:{}'.format(Y, m, d, H, M, S))      # TODO: figure out if np.datetime doesn't need to come from string... maybe a tuple?

# read in the data
# TODO: pandas merge for now. think this is slowing down when concatenating a lot of files
dfs = []
for fn in args.fns:
    df = pd.read_csv(fn, delim_whitespace=True, header=None, parse_dates={'datetime': list(range(6))},
                     index_col='datetime', date_parser=dt_parser, usecols=list(range(6))+[6, 7, 11, 12],
                     comment='#'
                     )
    # change names
    #df = df.rename(columns={6:'Bx', 7:'By', 11:'Ex', 12:'Ey'})
    df = df.rename(columns=dict(zip(args.col_order, ['Bx', 'By', 'Ex', 'Ey'])))
    dfs.append(df)

df = pd.concat(dfs, axis='rows')
df.sort_index(inplace=True)
df = df.resample('S').mean()
df.interpolate(limit=3, inplace=True)       # TODO: replace with missing sample limit

# need to replace with 0.0
df.fillna(0.0, inplace=True)

# ======================================================================================================================
# take dfts
dfts = AttrDict()
for col in df.columns:
    dfts[col] = np.fft.rfft(df[col].data)

f = np.fft.rfftfreq(n=len(df.index))

temp = AttrDict()
for interp_funct in Z:
    temp[interp_funct] = np.zeros(len(f), dtype=np.complex128)       # everything will be zero otherwise
    mask = (Z_ll <= f) & (f <= Z_ul)
    temp[interp_funct][mask] = Z[interp_funct](f[mask])
Z = temp

bpf = np.ones(len(f))
if args.lb != None:
    bpf[f < args.lb] = 0.0
if args.ub != None:
    bpf[args.ub < f] = 0.0

# ----------------------------------------------------------------------------------------------------------------------
# write to the dataframe
for E_comp, (m1, m2) in zip(['Ex', 'Ey'], [(Z.xx, Z.xy), (Z.yx, Z.yy)]):
    df[E_comp+'_bpf_der'] = np.fft.irfft(bpf*(m1*dfts.Bx+m2*dfts.By))

    # experimental filtered
    temp = dfts.Ex
    temp[~mask] = 0.0
    temp *= bpf
    df[E_comp+'_bpf'    ] = np.fft.irfft(temp)

# ----------------------------------------------------------------------------------------------------------------------
# TODO: should start 2 threads here... one for writing to screen/file, one for plotting data
# plot if want to
if args.show_figs:
    df[['Ex', 'Ey', 'Bx', 'By']].plot(subplots=True)
    df[['Ex_bpf', 'Ex_bpf_der']].plot(alpha=0.7)
    df[['Ey_bpf', 'Ey_bpf_der']].plot(alpha=0.7)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def write_output(fp):
    print('#'+str(args), file=fp)
    df.to_csv(fp, date_format='%Y-%m-%dT%H:%M:%SZ', float_format='%.2f', quoting=csv.QUOTE_NONE)          # % is the old way of doing string formating in python

if args.out_fn is None:
    write_output(sys.stdout)
else:
    with args.out_fn.open('w') as fp:
        write_output(fp)


