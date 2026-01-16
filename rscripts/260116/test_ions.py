from jica.io import read_data_from_name

# read ions data from "ions.npz"
ions = read_data_from_name("ions.npz")

# positive target
pos_target = ions.pos_target
print(pos_target)

# negative target
neg_target = ions.neg_target
print(neg_target)


# get all formulas and masses
print("\nALL IONS")
for f, m in zip(ions.formula, ions.mass):
    print(f"{f} : {m}")


# get all positive formulas and masses
print("\nPOSITIVE IONS")
for f, m, c in zip(ions.formula, ions.mass, ions.charge):
    # only positive charge
    if c > 0:
        print(f"{f}: {m}")

# get all negative formulas and masses
print("\nNEGATIVE IONS")
for f, m, c in zip(ions.formula, ions.mass, ions.charge):
    # only negative charge
    if c < 0:
        print(f"{f}: {m}")
