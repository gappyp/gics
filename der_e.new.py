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
from numpy import pi
import matplotlib.pyplot as plt
from gics.help import parse_edi, AttrDict, dt_parser, get_bounds, invl_intersection
from gics.help import lemi2dd
import csv
import time
from scipy.interpolate import interp1d

# ======================================================================================================================
hidden_init_fn = (Path(__file__).resolve().parent) / '.der_e'

parser = argparse.ArgumentParser(description='Derive electric field using impedance tensor and magnetic field data')
group = parser.add_mutually_exclusive_group()

parser.add_argument('-f', action='append', dest='fns', type=str, default=[], help='lemiMT time series file to include in processing')
group.add_argument('-i', action="store", dest="it_fn", type=Path, help='lemiMT .edi impedance tensor file')
parser.add_argument('-o', action="store", dest="out_fn", type=Path, help='Output file (if not specified will be outputted to stdout)')
parser.add_argument('--max-ts-gap', action="store", dest="max_ts_gap", type=int, default=0, help='Maximum number of missing samples that will linearly interpolate')
parser.add_argument('--lb', action="store", dest="lb", type=float, help='Lower f bound (Hz, linear scale) for bandpass filter (if not specified will use impedance tensor range)')
parser.add_argument('--ub', action="store", dest="ub", type=float, help='Upper f bound (Hz, linear scale) for bandpass filter (if not specified will use impedance tensor range)')
parser.add_argument('-c', action="store", dest="col_order", type=str, default='6,7,11,12', help='Column numbers of Bx,By,Ex,Ey in time series file (if not specified will use 6,7,11,12)')
parser.add_argument('-s', action="store", dest="signs", type=str, default='p,p,p,p', help='Signs to apply to Bx,By,Ex,Ey default (p,p,p,p)')
parser.add_argument('--figs', action="store_true", dest="show_figs", help='Plot figures of source and derived data')
parser.add_argument('--init-fn', action="store", dest="init_fn", type=Path, default=hidden_init_fn, help='Use a different initialization file')
group.add_argument('--sigma', action="store", dest="sigma", type=float, help='Sigma for half-space')
group.add_argument('--id', action="store", dest="id", type=str, help='3d model lat,long (e.g. 36.75S,143.25E)')

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

if args.it_fn:
    source_str = str(args.it_fn)
elif args.sigma:
    source_str = '{} ohm/m'.format(args.sigma)
    smpls = None
elif args.id:
    source_str = str(args.id)
    l2s = {'N':1.0, 'S':-1.0, 'E':1.0, 'W':-1.0}        # letter to sign
    args.id = [float(x[:-1])*l2s[x[-1]] for x in args.id.split(',')]
else:
    sys.exit('No impedance tensor source specified')        # TODO: make this print out to error

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
            sys.exit('Time series file "{}" does not exist'.format(single_file))        # TODO: to error
        temp.append(single_file)
args.fns = temp

# ======================================================================================================================
# change col order so can be used by pandas
args.col_order = [int(x) for x in args.col_order.split(',')]        # TODO: make these robust like the sign multiplier

# ----------------------------------------------------------------------------------------------------------------------
# take a peak at time-series file start and end time
temp = []
for fn in args.fns:
    temp.append((fn, get_bounds(fn)))
args.fns = temp         # args.fns now contains info about the bounds/timespan/interval of each file

# sort based on start of interval
args.fns = sorted(args.fns, key=lambda x: x[1][0])

# ----------------------------------------------------------------------------------------------------------------------
COLS_DT = ['Y', 'm', 'd', 'H', 'M', 'S']
COLS_BE = ['Bx', 'By', 'Ex', 'Ey']

# OLD...
# read in the data
COLS = ['Bx', 'By', 'Ex', 'Ey']
dfs = []
for fn, _ in args.fns:
    df = pd.read_csv(fn, delim_whitespace=True, header=None, parse_dates={'datetime': list(range(6))},
                     index_col='datetime', date_parser=dt_parser, usecols=list(range(6))+args.col_order)
    # change names
    df = df.rename(columns=dict(zip(args.col_order, COLS_BE)))
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
    dfts[col] = np.fft.rfft(df[col])

f = np.fft.rfftfreq(n=len(df.index))

# ======================================================================================================================
u0 = 1.25663706*(10**-6)                        # perm. of free space. from google
# don't pass 0 frequencies to this...
def gen_C_hs(f, sigma):
    #print('my sig is... {}'.format(sigma))
    # for half-space C = 1/q
    q = np.sqrt(1j*u0*sigma*2*pi*f)
    C = 1.0/q
    return C

# parse impedance tensor
if args.it_fn:
    Z, (Z_ll, Z_ul), smpls = parse_edi(args.it_fn)
elif args.sigma:
    Z_ll = f[1]
    Z_ul = f[-1]
    Z = AttrDict()
    Z.xx = lambda f: 0.0
    Z.yy = Z.xx
    Z.xy = lambda f: gen_C_hs(f, sigma=args.sigma)*(1j)*2*pi*f*(10**-3)         # negative power 3 must be for SI -> mv/(km*nt)???
    Z.yx = lambda f: -1.0*Z.xy(f)
    # liejun wanted to look @ some values
    #print(Z.xy(0.001))
    #print(Z.xy(0.01))
    #print(Z.xy(0.1))
elif args.id:
    ljw_model = pd.read_csv(Path(r"G:\python_projs\gics\respo_2777sites_TAS_edited.dat"), delim_whitespace=True, skiprows=8, header=None)
    ljw_model['f'] = 1/ljw_model[0]
    this_site = ljw_model.loc[(ljw_model[2] == args.id[0]) & (ljw_model[3] == args.id[1])]             # TODO: replace with site
    assert len(this_site) != 0      # invalid key
    # fill out smpls
    smpls = AttrDict()

    Z_ll = this_site.iloc[-1].f
    Z_ul = this_site.iloc[0].f
    Z = AttrDict()
    for comp in ['ZXX', 'ZXY', 'ZYX', 'ZYY']:
        this_comp = this_site.loc[this_site[7] == comp]
        Z[comp[1:].lower()] = interp1d(this_comp['f'], this_comp[8]+this_comp[8]*1j, kind='cubic')
        smpls[comp[1:].lower()] = this_comp[8]+this_comp[8]*1j
        smpls.f = this_comp.f

#print(Z_ll, Z_ul)
#sys.exit()
# ======================================================================================================================
temp = AttrDict()
for interp_funct in Z:
    temp[interp_funct] = np.zeros(len(f), dtype=np.complex128)       # everything will be zero otherwise
    mask = (Z_ll <= f) & (f <= Z_ul)
    temp[interp_funct][mask] = Z[interp_funct](f[mask])

    if args.show_figs:
        if smpls != None:
            pass
            plt.scatter(smpls.f, np.abs(smpls[interp_funct]), alpha=0.7, label=None)
        if args.sigma != None and (interp_funct in ['xx', 'yy']):continue       # bit of a hack lol
        plt.plot(f[mask], np.abs(temp[interp_funct][mask]), label=interp_funct, alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')


Z = temp

bpf = np.ones(len(f))


if args.lb == None:
    args.lb = f[1]           # try negative inf
if args.ub == None:
    args.ub = f[-1]          # TODO: need to set this to nquist freq

bpf[0] = 0.0        # zero out dc
bpf[f <= args.lb] = 0.0
bpf[args.ub <= f] = 0.0

# ----------------------------------------------------------------------------------------------------------------------
# write to the dataframe
for E_comp, (m1, m2) in zip(['Ex', 'Ey'], [(Z.xx, Z.xy), (Z.yx, Z.yy)]):
    df[E_comp+'_bpf_der'] = np.fft.irfft(bpf*(m1*dfts.Bx+m2*dfts.By))

    # experimental filtered
    temp = dfts[E_comp]
    temp[~mask] = 0.0
    temp *= bpf
    df[E_comp+'_bpf_orig'    ] = np.fft.irfft(temp)

# ----------------------------------------------------------------------------------------------------------------------
# TODO: should start 2 threads here... one for writing to screen/file, one for plotting data
# plot if want to
if args.show_figs:
    plt.legend()
    plt.xlabel('f (Hz)')
    plt.ylabel('|Z| (mV/(km*nT))')
    plt.title(source_str)
    # bandpass filter
    plt.axvspan(args.lb, args.ub, alpha=0.05)

    df[['Ex', 'Ey', 'Bx', 'By']].plot(subplots=True)
    df[['Ex_bpf_orig', 'Ex_bpf_der']].plot(alpha=0.7)
    df[['Ey_bpf_orig', 'Ey_bpf_der']].plot(alpha=0.7)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
def write_output(fp):
    print('# Z source: '+source_str, file=fp)
    print('# Z F range: '+'{} to {} Hz (inclusive)'.format(Z_ll, Z_ul), file=fp)
    print('# Bandpass filter range: {} to {} Hz (exclusive)'.format(args.lb, args.ub), file=fp)
    print('# DFT F bin range: {} to {} Hz'.format(f[1], f[-1]), file=fp)
    effective = invl_intersection((Z_ll, Z_ul), (args.lb, args.ub), (f[1], f[-1]))
    print('# Effective F range: {} to {} Hz'.format(*effective), file=fp)               # TODO: untested
    print('# Time-series files: {}'.format([str(fn) for (fn, _) in args.fns]), file=fp)
    print('# Signs: {}'.format(args.signs), file=fp)
    print('# t range: {} to {}'.format(df.index[0], df.index[-1]), file=fp)
    print('# Samples: {}'.format(len(df.index)), file=fp)
    df.to_csv(fp, date_format='%Y-%m-%dT%H:%M:%SZ', float_format='%.2f', quoting=csv.QUOTE_NONE)          # % is the old way of doing string formating in python

if args.out_fn is None:
    write_output(sys.stdout)
else:
    with args.out_fn.open('w') as fp:
        write_output(fp)