import sys
import scipy as sp
import matplotlib.pyplot as plt

batch_name = "default"
if len(sys.argv) >= 2:
  batch_name = sys.argv[1];

# reading tab separated data
data = sp.genfromtxt( "data/{}_training_data.tsv".format(batch_name),
                      delimiter="\t" )

in_coeffs = sp.genfromtxt( "data/{}_input_poly.tsv".format(batch_name),
                      delimiter="\t" )

out_coeffs = sp.genfromtxt( "data/{}_output_poly.tsv".format(batch_name),
                      delimiter="\t" )

# convert 2-dim array into two separate flat arrays
x = data[:, 0]
y = data[:, 1]

# scatter plot input data for revision
plt.scatter( x, y, s=10 )
plt.title( "Input/output polynomials and training data points" )
plt.xlabel( "x" )
plt.ylabel( "y" )
plt.grid( True, linestyle='-', color='0.75' )

in_poly = sp.poly1d(in_coeffs[::-1])
out_poly = sp.poly1d(out_coeffs[::-1])

fx = sp.linspace( -10, 10, 1000 )

plt.plot( fx, in_poly( fx ), linewidth=2 )
plt.plot( fx, out_poly( fx ), linewidth=2 )

plt.legend( [ "given polynomial", "result polynomial" ], loc="upper left" )
plt.show()
