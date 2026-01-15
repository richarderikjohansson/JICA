from jica.io import find_file_from_id
import numpy as np

i73 = np.load(find_file_from_id(id=73))
print(i73)
