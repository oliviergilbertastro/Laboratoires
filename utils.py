"""
Needed to make a lot of tables in the terminal and there is no support for it, so I made this program.
There are now various useful general functions in here, and some functions made specifically for the QPE vs TDE paper
"""

import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
import seaborn as sns
from tqdm import tqdm

def print_table(a, header=None, title=None, space_between_columns=2, space_between_rows=0, borders=1, header_color="yellow", border_color="grey", override_length=None):
    """
    Nicely print out a table

    a: array to print
    header: either an array of column names or a boolean. True -> first row will be header
    title: string, title that will be centered above the table
    space_between_columns: int, self-explanatory
    space_between_rows: int, self-explanatory
    """
    a = np.array(a, dtype=str)
    if type(header) == str or (type(header) == bool and header):
        pass
    elif (type(header) == bool and header == False) or header == None:
        header = None
    else:
        a = np.vstack((header, a))

    #Initialize the ascii characters to create the table depending on the borders parameter:
    if borders == None or borders == 0 or borders == False or borders == "none":
        characters = [" "," "," "," "," "," "]
    elif borders == "bold" or borders == 2:
        characters = ["═","║","╔","╗","╚","╝"]
    elif borders == 1 or borders == True or borders == "normal":
        characters = ["─","│","┌","┐","└","┘"]
    else:
        if type(borders) == str and len(borders) == 1:
            characters = [*(borders*6)]
        else:
            raise ValueError(f"Border style '{borders}' does not exist, use the keyword 'none', 'normal' or 'bold'.")
    
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    #Initialize the colors:
    #Header color
    if header_color == None or header_color == "grey":
        header_color = "0"
    elif type(header_color) == str:
        header_color = header_color.lower()
        if header_color in possible_colors:
            header_color = str(possible_colors.index(header_color)+30)
        else:
            print(f"Color '{header_color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            header_color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    #Borders color
    if border_color == None or border_color == "grey":
        border_color = "0"
    elif type(border_color) == str:
        border_color = border_color.lower()
        if border_color in possible_colors:
            border_color = str(possible_colors.index(border_color)+30)
        else:
            print(f"Color '{border_color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            border_color = "0"
    else:
        raise ValueError("Parameter 'border_color' needs to be a string.")

    for i in range(len(characters)):
        characters[i] = f"\x1b[{border_color}m{characters[i]}\x1b[0m"

    #Replace (None) elements with "-":
    a[a == "None"] = "-"

    #Get longest string in each column:
    column_maxes = []
    vfunc = np.vectorize(lambda x: len(x))
    a_lens = vfunc(a)
    for i in range(a.shape[1]):
        column_maxes.append(np.max(a_lens[:,i]))
    if override_length != None:
        column_maxes = override_length
    total_length = np.sum(column_maxes)+(len(column_maxes)-1)*space_between_columns #To include spaces between each column

    #Actually start printing table:
    top_and_bottom_bounds = (characters[2]+characters[0]*(total_length+2)+characters[3], characters[4]+characters[0]*(total_length+2)+characters[5])
    print()
    usable_length = total_length+4
    if title != None:
        title = floor((usable_length-len(title))/2)*" "+title+ceil((usable_length-len(title))/2)*" "
        print(f"\x1b[{header_color}m{title}\x1b[0m")
    print(top_and_bottom_bounds[0])
    #Print each row:
    for row in range(a.shape[0]):
        row_string = ""
        for column in range(a.shape[1]):
            row_string += a[row, column] + " "*(column_maxes[column]-a_lens[row,column])
            if column < a.shape[1]-1:
                row_string += " "*space_between_columns
        if row == 0 and header != None:
            row_string = f"\x1b[{header_color}m{row_string}\x1b[0m"
        row_string = f"{characters[1]} {row_string} {characters[1]}"
        if row != (a.shape[0]-1):
            row_string += f"\n{characters[1]} {' '*(total_length)} {characters[1]}"*space_between_rows
            if row == 0 and header != None:
                row_string += f"\n{characters[1]} {' '*(total_length)} {characters[1]}"
        print(row_string)
    print(top_and_bottom_bounds[1])
    print()





def toLog(a, inverse=False):
    """Convert array of data and uncertainties to/from log base"""
    if not inverse:
        a = np.array(a)
        try:
            data, lo, hi = a[:,0], a[:,1], a[:,2]
        except:
            data, lo, hi = a
        lo = np.abs(lo/(data*np.log(10)))
        hi = np.abs(hi/(data*np.log(10)))
        data = np.log10(data)
        return np.array([data, hi, lo]).T
    else:
        a = np.array(a)
        data, lo, hi = 10**a[:,0], 10**a[:,0]*np.log(10)*a[:,1], 10**a[:,0]*np.log(10)*a[:,2]
        return np.array([data, hi, lo]).T

def SDSS_objid_to_values(objid):

    # Determined from http://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID

    bin_objid = bin(objid)
    bin_objid = bin_objid[2:len(bin_objid)]
    bin_objid = bin_objid.zfill(64)

    empty = int( '0b' + bin_objid[0], base=0)
    skyVersion = int( '0b' + bin_objid[1:4+1], base=0)
    rerun = int( '0b' + bin_objid[5:15+1], base=0)
    run = int( '0b' + bin_objid[16:31+1], base=0)
    camcol = int( '0b' + bin_objid[32:34+1], base=0)
    firstField = int( '0b' + bin_objid[35+1], base=0)
    field = int( '0b' + bin_objid[36:47+1], base=0)
    object_num = int( '0b' + bin_objid[48:63+1], base=0)

    return skyVersion, rerun, run, camcol, field, object_num

from astropy.coordinates import SkyCoord
import astropy.units as u
def get_smallest_sep(pos, ras, decs):
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg)
    c2 = SkyCoord(ras*u.deg, decs*u.deg)
    sep = (c1.separation(c2)).arcsec
    smallest_sep = min(sep)
    index = list(sep).index(smallest_sep)
    return index, smallest_sep


def get_smallest_sep_v2(pos, ras, decs):
    c1 = SkyCoord(pos[0]*u.deg, pos[1]*u.deg)
    c2 = SkyCoord(ras*u.deg, decs*u.deg)
    sep = (c1.separation(c2)).arcsec
    idx = (sep).argmin()
    return idx, sep[idx]



def print_color(message, color="yellow", **kwargs):
    """print(), but with a color option"""
    possible_colors = ["black","red","green","yellow","blue","magenta","cyan","white"]
    if color == None or color == "grey":
        color = "0"
    elif type(color) == str:
        color = color.lower()
        if color in possible_colors:
            color = str(possible_colors.index(color)+30)
        else:
            print(f"Color '{color}' not implemented, defaulting to grey.\nPossible colors are: {['grey']+possible_colors}")
            color = "0"
    else:
        raise ValueError(f"Parameter 'header_color' needs to be a string.")
    print(f"\x1b[{color}m{message}\x1b[0m", **kwargs)



import copy

def recombine_arrays(data, lo, hi):
    """
    Recombine 2D arrays into a single 3D array
    data: array containing the data
    lo: array containing the lower uncertainties
    hi: array containing the high uncertainties
    """

    assert data.shape == lo.shape == hi.shape

    new_array = np.zeros((*(data.shape),3))
    new_array[:,:,0] = data
    new_array[:,:,1] = lo
    new_array[:,:,2] = hi

    return new_array

def cut_from_catalog(catalog, index, bounds, verbose=False):
    """
    catalog: numpy array
    index: int of index of parameter column which is under study here
    bounds: tuple of bound which we want to keep

    returns: numpy array with only remaining objects
    """
    catalog = np.array(catalog)
    if bounds[0] == None:
        good_indices_lo = catalog[:,index] == catalog[:,index]
    else:
        good_indices_lo = catalog[:,index] >= bounds[0]
    if bounds[1] == None:
        good_indices_hi = catalog[:,index] == catalog[:,index]
    else:
        good_indices_hi = catalog[:,index] <= bounds[1]
    good_indices = []
    for i in range(len(good_indices_lo)):
        good_indices.append(good_indices_lo[i] and good_indices_hi[i])
    cut_catalog = catalog[good_indices]
    if verbose:
        print(f"\x1b[31m{catalog.shape[0]-cut_catalog.shape[0]} objects cut\x1b[0m")
        print(f"\x1b[32m{cut_catalog.shape[0]} objects remaining\x1b[0m")
    return cut_catalog

def mergeCatalogs_withObjIDs(cat1,cat2,columnsToAdd=[0]):
    """
    Merge catalog1 with catalog2 assuming their first columns are the objIDs
    """
    good_indices = []
    properties_toAdd = []
    for i in range(len(columnsToAdd)):
        properties_toAdd.append([])
    for i in tqdm(range(len(cat1[:,0]))):
        try:
            index = list(cat2[:,0]).index(cat1[i,0])
            good_indices.append(i)
            for k in range(len(columnsToAdd)):
                properties_toAdd[k].append(cat2[index,columnsToAdd[k]])
            #print(f"{cat1[i,0]} vs {cat2[index,0]}")
        except:
            pass
    cat1 = cat1[[good_indices],:][0]
    for i in range(len(columnsToAdd)):
        cat1 = np.vstack((cat1.T, np.array(properties_toAdd[i]))).T
    return cat1

def cut_from_array(a, indices):
    indices = [(i if i >= 0 else len(a)+i) for i in indices]
    return a[[(i not in indices) for i in range(len(a))]]



def add_0_uncertainties(a, value=0):
    """
    a: array of shape (n,)
    value: value to set uncertainties (default=0)

    Function to add null uncertainties to an array of data.
    Takes in an array of shape (n,) and returns an array of shape (n,3).
    E.g. 
    [3,4,5,6] -> [[3,0,0],[4,0,0],[5,0,0],[6,0,0]]
    
    """
    a = np.array(a)
    try:
        if a.shape[1] == 3:
            print("No uncertainties added, already containing uncertainties.")
            return a
    except:
        placeholder = np.ones((a.shape[0],3))*value
        placeholder[:,0] = a
        a = placeholder
        return a


import time
if __name__ == "__main__":

    start_time = time.time()
    print(get_smallest_sep([2.36286871e+02, -5.18003056e-01], [2.36247096e+02,2.36286871e+02,2.36336501e+02,2.15640297e+02], [-4.75263889e-01,-5.18003056e-01,-4.89098889e-01,1.06889111e+00]))
    print("\x1b[33m: --- %s seconds ---\x1b[0m" % (time.time() - start_time))
    start_time = time.time()
    print(get_smallest_sep_v2([2.36286871e+02, -5.18003056e-01], [2.36247096e+02,2.36286871e+02,2.36336501e+02,2.15640297e+02], [-4.75263889e-01,-5.18003056e-01,-4.89098889e-01,1.06889111e+00]))
    print("\x1b[33m: --- %s seconds ---\x1b[0m" % (time.time() - start_time))