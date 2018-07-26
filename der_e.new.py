"""
* derive electric field using impedance tensor and magnetic field data
"""

import argparse
import sys
import pathlib
from pathlib import Path
import shlex
from pprint import pprint
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gics.help import parse_edi, AttrDict, parse_intvl, reverse_readline      # helper function and classes
import csv
import time

# ======================================================================================================================
hidden_init_fn = (Path(__file__).resolve().parent) / '.der_e'

parser = argparse.ArgumentParser(description='Derive electric field using impedance tensor and magnetic field data')

parser.add_argument('-f', action='append', dest='fns', type=str, default=[], help='lemiMT time series file to include in processing')
parser.add_argument('-i', action="store", dest="it_fn", type=Path, help='lemiMT .edi impedance tensor file')
parser.add_argument('-o', action="store", dest="out_fn", type=Path, help='Output file (if not specified will be outputted to stdout)')
#parser.add_argument('--interval', action="store", dest="intvl", type=str, help='Interval (if data isn\'t provided, will be zero padded). If this option isn\'t specified, will determine from time series files')
parser.add_argument('--max-ts-gap', action="store", dest="max_ts_gap", type=int, default=0, help='Maximum number of missing samples that will linearly interpolate')
parser.add_argument('--lb', action="store", dest="lb", type=float, help='Lower f bound (Hz, linear scale) for bandpass filter (if not specified will use impedance tensor range)')
parser.add_argument('--ub', action="store", dest="ub", type=float, help='Upper f bound (Hz, linear scale) for bandpass filter (if not specified will use impedance tensor range)')
parser.add_argument('-c', action="store", dest="col_order", type=str, default='6,7,11,12', help='Column numbers of Bx,By,Ex,Ey in time series file (if not specified will use 6,7,11,12)')
parser.add_argument('-s', action="store", dest="signs", type=str, default='p,p,p,p', help='Signs to apply to Bx,By,Ex,Ey')
parser.add_argument('--figs', action="store_true", dest="show_figs", help='Plot figures of source and derived data')
parser.add_argument('--init-fn', action="store", dest="init_fn", type=Path, default=hidden_init_fn, help='Use a different initialization file')

# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args(sys.argv[1:])

if args.init_fn.is_file():
    with args.init_fn.open('r') as fp:
        init_args = fp.read()
    init_args = shlex.split(init_args, comments=True)
elif args.init_fn == hidden_init_fn:
    init_args = []
else:
    sys.exit('Impedance tensor file "{}" does not exist'.format(args.init_fn.absolute()))

all_args = init_args+sys.argv[1:]
args = parser.parse_args(all_args)

# ----------------------------------------------------------------------------------------------------------------------
# deal with glob style filenames and check if files exist
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
            sys.exit('Time series file "{}" does not exist'.format(single_file))
        temp.append(single_file)
args.fns = temp

# ======================================================================================================================
# parse impedance tensor
Z, (Z_ll, Z_ul) = parse_edi(args.it_fn)         # TODO: otherwise half-space... and will use max possible bounds if not specified

# parse interval
# args.intvl = parse_intvl(args.intvl)          # TODO: function needs completing
args.intvl = None                               # will return None if not specified...  # TODO: need parse_intvl to accept None and return it

# change col order so can be used by pandas
args.col_order = [int(x) for x in args.col_order.split(',')]

# ======================================================================================================================
# from https://stackoverflow.com/questions/12269528/using-python-pandas-to-parse-csv-with-date-in-format-year-day-hour-min-sec
def dt_parser(Y, m, d, H, M, S):
    return np.datetime64('{}-{}-{}T{}:{}:{}'.format(Y, m, d, H, M, S))      # TODO: figure out if np.datetime doesn't need to come from string... maybe a tuple?

# ----------------------------------------------------------------------------------------------------------------------
# take a peak at time-series file start and end time
# TODO: compare methods!!! (just reading in and discarding or peek @ back)

b4 = time.time()
# get date of first sample in file
with args.fns[0].open('r') as fp:
    for line in fp:
        if line.isspace(): continue
        else:
            # get datetime so can allocate
            line_sws = line.split()
            fn_start_dt = '{}-{}-{}T{}:{}:{}'.format(*line_sws[:7])
            break

# get date of last sample in file
for line in reverse_readline(args.fns[0]):
    if line.isspace(): continue
    else:
        # get datetime so can allocate
        line_sws = line.split()
        fn_end_dt = '{}-{}-{}T{}:{}:{}'.format(*line_sws[:7])
        break
else:
    pass
    # TODO:

print(fn_start_dt, fn_end_dt)
print(time.time()-b4)

# **********************************************************************************************************************
b4 = time.time()
df = pd.read_csv(args.fns[0], delim_whitespace=True, header=None, parse_dates={'datetime': list(range(6))},
                 index_col='datetime', date_parser=dt_parser, usecols=list(range(6))+[6, 7, 11, 12],
                 comment='#'
                 )
print(df.iloc[0], df.iloc[-1])
print(time.time()-b4)

sys.exit()

# ----------------------------------------------------------------------------------------------------------------------
# read in the data
COLS = ['Bx', 'By', 'Ex', 'Ey']
dfs = []
for fn in args.fns:
    df = pd.read_csv(fn, delim_whitespace=True, header=None, parse_dates={'datetime': list(range(6))},
                     index_col='datetime', date_parser=dt_parser, usecols=list(range(6))+[6, 7, 11, 12],
                     comment='#'        # TODO: think shouldn't support this comment
                     )
    # change names
    #df = df.rename(columns={6:'Bx', 7:'By', 11:'Ex', 12:'Ey'})
    df = df.rename(columns=dict(zip(args.col_order, COLS)))
    dfs.append(df)

df = pd.concat(dfs, axis='rows')
df.sort_index(inplace=True)
df = df.resample('S').mean()
df.interpolate(limit=3, inplace=True)       # TODO: replace with missing sample limit

# multiply by signs
args.signs = args.signs.split(',')
if [True if x in ['p', 'n'] else False for x in args.signs] != 4*[True]:
    raise ValueError('-s argument invalid')

l2m = {'p':1.0, 'n':-1.0}       # letter to multiplier
for col, sign in zip(COLS, args.signs):
    df[col] = l2m[sign]*df[col]

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
    temp = dfts[E_comp]
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
    pass
    #write_output(sys.stdout)
else:
    with args.out_fn.open('w') as fp:
        write_output(fp)


