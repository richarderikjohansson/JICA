from molmass import Formula
import numpy as np
from jica.io import get_datadir

ions = np.loadtxt("ions.txt", dtype=str)
targets = ["CH3Br-", "PC2H5+"]


mass = []
charge = []

for ion in ions:
    f = Formula(ion)
    mass.append(getattr(f, "mass"))
    charge.append(getattr(f, "charge"))

tp = Formula("PC2H5+")
tn = Formula("CH3Br-")

pos_target = np.array(["PC2H5+", tp.mass])
neg_target = np.array(["CH3Br-", tn.mass])

np.savez_compressed(
    get_datadir() / "ions.npz",
    formula=ions,
    mass=mass,
    charge=charge,
    pos_target=pos_target,
    neg_target=neg_target,
)
