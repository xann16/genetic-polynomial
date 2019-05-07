import sys
import scipy as sp
import matplotlib.pyplot as plt

batch_name = "default"

if len(sys.argv) >= 3:
  batch_name = sys.argv[2];

# reading tab separated data
data = sp.genfromtxt( "data/{}_progress_data.tsv".format(batch_name),
                      delimiter="\t" )

# convert 2-dim array into two separate flat arrays
x = data[:, 0]
err = data[:, 1]
avg_err = data[:, 2]

# scatter plot input data for revision
plt.title( "Training process error progress" )
plt.xlabel( "Generations" )
plt.ylabel( "Error" )
plt.grid( True, linestyle='-', color='0.75' )
plt.autoscale( tight=True )

plt.plot( x, err , linewidth=2 )
# plt.plot( x, sp.log10( err ), linewidth=2 )
plt.plot( x, avg_err , linewidth=2 )
# plt.plot( x, sp.log10( avg_err ), linewidth=2 )

plt.legend( [ "error of individual with best fitness",
              "average error of entire population" ],
            loc="upper left" )
plt.show()
