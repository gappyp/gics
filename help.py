"""
* helper function and classes for der_e.py
"""

from collections import defaultdict as dd
from scipy.interpolate import interp1d
from pprint import pprint
import sys
import numpy as np
import datetime
import os

# ======================================================================================================================
# from https://stackoverflow.com/questions/12269528/using-python-pandas-to-parse-csv-with-date-in-format-year-day-hour-min-sec
def dt_parser(Y, m, d, H, M, S):
    return np.datetime64('{}-{}-{}T{}:{}:{}'.format(Y, m, d, H, M, S))      # TODO: figure out if np.datetime doesn't need to come from string... maybe a tuple?

# ======================================================================================================================
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# ======================================================================================================================
def parse_edi(fna):
    vals = dd(list)
    with fna.open('r') as fp:
        while True:
            try:
                line = next(fp)
            except StopIteration:
                break
            for header in ['>FREQ', '>ZXXR', '>ZXXI', '>ZXYR', '>ZXYI', '>ZYXR', '>ZYXI', '>ZYYR', '>ZYYI']:
                if header in line:
                    # read until encounter new line
                    while True:
                        line = next(fp)
                        if line == '\n' or line[0] == '>':
                            break
                        else:
                            line_sw = line.split()
                            vals[header] += [eval(val) for val in line_sw]

    # store vals with better keys and create complex number types
    #f_smpls = [np.log10(x) for x in vals['>FREQ NFREQ']]       # samples provided by lemigraph. in log scale
    f_smpls = vals['>FREQ']       # samples provided by lemigraph. in linear scale
    Z_smpls = AttrDict()
    Z_smpls['xx'] = [(rl+img*1j) for rl, img in zip(vals['>ZXXR'], vals['>ZXXI'])]
    Z_smpls['xy'] = [(rl+img*1j) for rl, img in zip(vals['>ZXYR'], vals['>ZXYI'])]
    Z_smpls['yx'] = [(rl+img*1j) for rl, img in zip(vals['>ZYXR'], vals['>ZYXI'])]
    Z_smpls['yy'] = [(rl+img*1j) for rl, img in zip(vals['>ZYYR'], vals['>ZYYI'])]

    # assert Z_smpls are the same length
    assert len(set([len(smpls) for smpls in list(Z_smpls.values())+[f_smpls]])) == 1

    # for some reason frequency and values are decending. reverse it
    f_smpls.reverse()
    for key, val in Z_smpls.items():
        val.reverse()

    # assuming if it's missing in 1 component, it will be missing in all
    tempZ = AttrDict()
    tempZ.xx = []; tempZ.xy = []; tempZ.yx = []; tempZ.yy = []
    tempf = list()
    for f, xx, xy, yx, yy in zip(f_smpls, *Z_smpls.values()):
        if 1e+32+1e+32j in [xx, xy, yx, yy]:
            continue
        else:
            tempf.append(f)
            tempZ.xx.append(xx); tempZ.xy.append(xy); tempZ.yx.append(yx); tempZ.yy.append(yy)
    Z_smpls = tempZ
    f_smpls = tempf

    # need to get cubic splines now
    Z_css = AttrDict()
    for key, val in Z_smpls.items():
        Z_css[key] = interp1d(f_smpls, val, kind='cubic')

    Z_smpls['f'] = f_smpls

    return Z_css, (f_smpls[0], f_smpls[-1]), Z_smpls

# ======================================================================================================================
# from https://stackoverflow.com/questions/2301789/read-a-file-in-reverse-order-using-python
def reverse_readline(filename, buf_size=8192):
    """a generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # the first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # if the previous chunk starts right from the beginning of line
                # do not concact the segment to the last line of new chunk
                # instead, yield the segment first
                if buffer[-1] is not '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if len(lines[index]):
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment

# ======================================================================================================================
def get_bounds(fn):
    # get date of first sample in file
    with fn.open('r') as fp:
        for line in fp:
            if line.isspace(): continue
            else:
                # get datetime so can allocate
                line_sws = line.split()
                fn_start_dt = dt_parser(*line_sws[:6])
                break

    # get date of last sample in file
    for line in reverse_readline(fn):
        if line.isspace(): continue
        else:
            # get datetime so can allocate
            line_sws = line.split()
            fn_end_dt = dt_parser(*line_sws[:6])
            break

    return fn_start_dt, fn_end_dt

# ======================================================================================================================
# lemi timeseries position format to signed decimal degrees
# p1 is DDDMM.MMMM, p2 is N,S,E,W
l2m = {'N':1.0, 'S':-1.0, 'E':1.0, 'W':-1.0}
def lemi2dd(p1, p2):
    # TODO: older version is probably good enough and faster. evenutally profile and decide what to use (also try get 'S10' method working)
    """
    # old version using float
    # issues with precision
    mins = '{:.30f}'.format(int(p1)%100+p1%1)
    degs = int(p1/100)
    """

    # new version having strings passed
    #print(p1, p2)
    whole, dec = p1.split('.')
    deg, min = whole[:-2], whole[-2:]

    return (float(deg)+float('{}.{}'.format(min, dec))/60.0)*l2m[p2]

def invl_intersection(*args):
    ls, us = zip(*args)
    return (max(ls), min(us))