"""
script to plot results of V2V exchange location heat-map
currently on C225
25.01.2k19
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

CITY_NAME = 'SF'
v2v_heatmap_data_filepath = '/home/toshiba/ABMTANET/v2v_heat_maps/SF/'
numfolded_days = [1,2,4,6,9]

sub_plot_num_list = [111,122,133,214,225]
j = 0
#numfolded_days = 1
total_v2v_exchanges_list = []
v2v_heatmap_results_dict = dict()

for data in numfolded_days:

    v2v_heatmap_data_filename = ('sf_%i_days_v2v_heatmap_data.pickle' % data)

    with open((v2v_heatmap_data_filepath+v2v_heatmap_data_filename),'rb') as handle0:
        v2v_heatmap_array = pickle.load(handle0)

    total_real_v2v_exchanges = 0
    for i in range(v2v_heatmap_array.shape[0]):
        total_real_v2v_exchanges += np.sum(v2v_heatmap_array[i])


    v2v_heatmap_results_dict[data] = v2v_heatmap_array
    total_v2v_exchanges_list.append(total_real_v2v_exchanges)

    """
    fig, ax1 = plt.subplots(1,1)
    c = ax1.pcolor(v2v_heatmap_array)
    ax1.set_title('real... V2V Exchange Heat-map, %s, total: %i, NUM of folded days: %i' % (CITY_NAME, int(total_real_v2v_exchanges), data))
    fig.tight_layout()
    plt.show()


    """

    #ax1 = plt.subplot(sub_plot_num_list[j])
    #plt.imshow(v2v_heatmap_array/total_real_v2v_exchanges, cmap=plt.cm.BuPu_r)


    handle0.close()

    j +=1


#cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#plt.colorbar(cax=cax)
#plt.show()


plt.subplot(231)

ax1 = plt.subplot()






