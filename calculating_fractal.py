import matplotlib.pyplot as plt
import porespy as ps
ps.visualization.set_mpl_style()

im = ps.generators.sierpinski_foam(4, 5)
plt.imshow(im)


data = ps.metrics.boxcount(im)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('box edge length')
ax1.set_ylabel('number of boxes spanning phases')
ax2.set_xlabel('box edge length')
ax2.set_ylabel('slope')
ax2.set_xscale('log')
ax1.plot(data.size, data.count,'-o')
ax2.plot(data.size, data.slope,'-o')

plt.show()
