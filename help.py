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

    # for some reason frequency and values are decending. reverse it
    f_smpls.reverse()
    for key, val in Z_smpls.items():
        val.reverse()

    # need to get cubic splines now
    Z_css = AttrDict()
    for key, val in Z_smpls.items():
        Z_css[key] = interp1d(f_smpls, val, kind='cubic')

    return Z_css, (f_smpls[0], f_smpls[-1])

# ======================================================================================================================
iso_fmt = '%Y-%m-%dT%H:%M:%S'

# TODO: this is going to take some time to work on... try get a version out where this feature can come later (i.e. so data derived interval is only supported)
# parse the --interval option given to der_e.py from command line
# I intend it to be iso8601 format, but this function mightn't fully comply with it
# https://en.wikipedia.org/wiki/ISO_8601#Time_intervals
# this should hopefully parse time intervals expresses as 1,2,3 from the above link
def parse_intvl(intvl):
    if not isinstance(intvl, str):
        raise TypeError('str parsing only supported')       # TODO: confirm this is the way to set exception message

    intvl_ss = intvl.split('/')     # split slash

    if len(intvl_ss) != 2:
        raise ValueError("Interval needs one '/' character")

    p1, p2 = intvl_ss

    try:
        p1 = datetime.datetime.strptime(p1, iso_fmt)
    except ValueError:
        pass

    print(p1)

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