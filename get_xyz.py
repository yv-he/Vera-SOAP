"""
19.05.22
@tcnicholas
"""

import numpy as np

def main():
    """
    Read in all zinc positions from txt file and print out as XYZ file.
    """
    
    # extract all positions.
    allPositions = np.loadtxt(  "zn_coordinates.txt",
                                skiprows=4,
                                dtype=str)[:,2:].astype(np.float64)
    # write xyz file.
    # first line: number of atoms.
    # second line: Properties list (separated by :).
    # each atom line: symbol, *coordinates.
    outstr = f"{allPositions.shape[0]}\n"
    outstr += f'Properties=species:S:1:positions:R:3 pbc="F F F"\n'
    for atom in allPositions:
        outstr += "Zn{:>15.5f}{:>15.5f}{:>15.5f}\n".format(*list(atom))
        
    # write to file.
    with open("zn_coords.xyz", "w") as f:
        f.write(outstr)
    
    
    
if __name__ == "__main__":
    main()
