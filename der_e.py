"""
* TODO: thing in argparse that lets specify valid options. will be good for e.g. window functions
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
from gics.help import parse_edi
from geomag.kdu_viewer.help import AttrDict

pd.options.display.max_rows = 1000

init_fn = Path(r'G:\python_projs\gics\.der_e')      # TODO: change so can use different. still like the @thing... maybe use and fix it eventually
with init_fn.open('r') as fp:
    init_args = fp.read()

parser = argparse.ArgumentParser(description='Derive electric fields')

parser.add_argument('-f', action='append',
                    dest='fns',
                    type=str,
                    default=[],
                    help='time series file to include in processing')
parser.add_argument('--filelist', action='append',              # TODO: don't know if need this... might be better to get the recursive '@' cmd options working
                    dest='filelists',
                    default=[],
                    help='filename of a list of time series files to include in processing')
parser.add_argument('-a', action="store", dest="hs_alpha", type=float, help='half-space alpha value')      # TODO: can put ranges on this e.g. choices=[Range(0.0, 1.0)]. see https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
parser.add_argument('-i', action="store", dest="it_fn", type=Path, help='impedance tensor filename')      # TODO: can put ranges on this e.g. choices=[Range(0.0, 1.0)]. see https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
parser.add_argument('-o', action="store", dest="out_fn", type=Path, help='output filename (if not specified will be outputted to stdout)')
parser.add_argument('-w', action="store", dest="wnd_funct", type=str, help='windowing function (if not specified will be outputted to stdout)')
parser.add_argument('--max-ts-gap', action="store", dest="max_ts_gap", type=int, help='maximum number of missing samples that will interpolate')
parser.add_argument('--ts-gap-interp-method', action="store", dest="ts_gap_interp_method", type=str, help='interpolation method to use for time series gaps')       # TODO: these can't be optional
parser.add_argument('--lb', action="store", dest="lb", type=float, help='lower bound for bandpass filter (log scale)')       # TODO: these can't be optional
parser.add_argument('--ub', action="store", dest="ub", type=float, help='upper bound for bandpass filter (log scale)')       # TODO: these can't be optional
# TODO: zero pad to a certain number of days? or a certain number of samples?

init_args = shlex.split(init_args, comments=True)

args = parser.parse_args(init_args+sys.argv[1:])

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

#pprint(args)
# ======================================================================================================================
# parse impedance tensor
Z, (Z_ll, Z_ul) = parse_edi(args.it_fn)

# ======================================================================================================================
# from https://stackoverflow.com/questions/12269528/using-python-pandas-to-parse-csv-with-date-in-format-year-day-hour-min-sec
def dt_parser(Y, m, d, H, M, S):
    return np.datetime64('{}-{}-{}T{}:{}:{}'.format(Y, m, d, H, M, S))      # TODO: figure out if np.datetime doesn't need to come from string... maybe a tuple?

# read in the data
# TODO: pandas merge for now. optimize if too slow
dfs = []
for fn in args.fns:
    df = pd.read_csv(fn, delim_whitespace=True, header=None, parse_dates={'datetime': list(range(6))},
                     index_col='datetime', date_parser=dt_parser, usecols=list(range(6))+[6, 7, 11, 12],
                     comment='#'
                     )
    # change names
    df = df.rename(columns={6:'Bx', 7:'By', 11:'Ex', 12:'Ey'})
    dfs.append(df)

df = pd.concat(dfs, axis='rows')
df.sort_index(inplace=True)
df = df.resample('S').mean()
df.interpolate(limit=3, inplace=True)       # TODO: replace with missing sample limit

# ======================================================================================================================
if args.wnd_funct != None:
    exec('wnd_funct = np.{}(len(df.index))'.format(args.wnd_funct))
else:
    wnd_funct = np.ones(len(df.index))

# take dfts
dfts = AttrDict()
for col in df.columns:
    dfts[col] = np.fft.rfft(wnd_funct*df[col].data)

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

der_Ex_f = bpf*(Z.xx*dfts.Bx+Z.xy*dfts.By)
der_Ex_t = np.fft.irfft(der_Ex_f)

exp_Ex_f = dfts.Ex
exp_Ex_f[~mask] = 0.0
exp_Ex_f *= bpf
exp_Ex_t = np.fft.irfft(exp_Ex_f)

plt.plot(der_Ex_t)
plt.plot(exp_Ex_t)

plt.ylim(-20, 20)
plt.show()