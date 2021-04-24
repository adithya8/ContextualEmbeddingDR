import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
mpl.rcParams['svg.hashsalt'] = 42
np.random.seed(42)

#print (plt.style.available)
#https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
styles = ['seaborn-notebook', '_classic_test', 'seaborn-pastel', 'seaborn-talk', 'seaborn', 'seaborn-poster', \
        'seaborn-deep', 'seaborn-ticks', 'seaborn-paper', 'grayscale', 'seaborn-dark-palette',\
        'seaborn-whitegrid', 'classic', 'ggplot', 'seaborn-colorblind', 'seaborn-bright', 'bmh',\
        'Solarize_Light2', 'seaborn-white', 'fast', 'dark_background', 'fivethirtyeight', 'seaborn-dark', \
        'seaborn-muted', 'seaborn-darkgrid', 'tableau-colorblind10']
plt.style.use('tableau-colorblind10')
data = pd.read_csv("./nvsk_std_minmax_.csv")

n = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
ks = [16, 32, 64, 128, 256, 512]
task_grps = [('age',), ('gen',), ('gen2',), ('ext',), ('ope',), ('bsag',), ('sui',)]
task_grps_name = ["Age", "Gender", "Gender2", "Ext", "Ope", "BSAG", "Sui"]
task_grps_file = ["Age", "Gender", "Gender2", "Ext", "Ope", "BSAG", "Sui"]
task_grps_label = ["Age", "Gender", "Gender2", "Ext", "Ope", "BSAG", "Sui"]
task_grps_score = ["Pearson-r", "macro-f1", "macro-f1", "Pearson-r*", "Pearson-r*", "Pearson-r*", "macro-f1"]
task_color_dict = {
    "age": ["#ff3333", "#33beff", "#33ff42", 1],
    "gen": ["#00a630", "#4e00a6", "#a60000", 1],
    "gen2": ["#ffd800","#00fffb", "#ff00a6", 1],
    "ext": ["", "", "", np.sqrt(0.70*0.77)],
    "ope": ["", "", "", np.sqrt(0.70*0.77)],
    "bsag": ["", "", "", np.sqrt(0.70*0.77)],
    "sui": ["", "", "", 1]
}

for grp_no in range(len(task_grps)):
        for i in n[4:]:
                temp_data = data[(data.Task.isin(task_grps[grp_no])) & (data.N >=1000 )]
                y_max, y_min = [], []
                for task in temp_data.Task.unique():
                        
                        temp = temp_data[temp_data.Task == task]
                        y_max_nodr = np.max(temp_data.values[:, 14:15] + (temp_data.values[:, 15:16]/np.sqrt(10)))
                        y_min_nodr = np.min(temp_data.values[:, 14:15] - (temp_data.values[:, 15:16]/np.sqrt(10)))
                        if (task.lower() == "sui"):
                                y_max_temp = np.max(temp_data.values[:, 2:11:2] + (temp_data.values[:, 3:12:2]/np.sqrt(10)))
                                y_min_temp = np.min(temp_data.values[:, 2:11:2] - (temp_data.values[:, 3:12:2]/np.sqrt(10)))
                        else:
                                y_max_temp = np.max(temp_data.values[:, 2:13:2] + (temp_data.values[:, 3:14:2]/np.sqrt(10)))
                                y_min_temp = np.min(temp_data.values[:, 2:13:2] - (temp_data.values[:, 3:14:2]/np.sqrt(10)))

                        y_max_temp = np.max([y_max_temp, y_max_nodr])
                        y_min_temp = np.min([y_min_temp, y_min_nodr])
                        y_max.append(y_max_temp)
                        y_min.append(y_min_temp)
                
                y_max = np.mean(y_max)
                y_min = np.mean(y_min)
                

                temp_data = data[(data.N == i) & (data.Task.isin(task_grps[grp_no]))]                
                fig, ax = plt.subplots(figsize=(15, 15))
                #for j in temp_data.values:
                j = temp_data.values
                #print ((np.mean(j[:, 2:13:2], axis=0).reshape(-1,1)/1).flatten().tolist())
                y = (np.mean(j[:, 2:13:2], axis=0)/task_color_dict[j[0,0]][-1]).tolist()
                y_std = ((np.mean(j[:, 3:14:2], axis=0)/task_color_dict[j[0,0]][-1])/np.sqrt(10)).tolist()
                y_nodr = (np.mean(j[:, 14:15], axis=0)/task_color_dict[j[0,0]][-1]).tolist()
                y_nodr_std = (np.mean(j[:, 15:16], axis=0)/task_color_dict[j[0,0]][-1]/np.sqrt(10)).tolist()
                y_ci = (np.mean(j[:, 16:17], axis=0)/task_color_dict[j[0,0]][-1]).tolist()
                y_max = y_max/task_color_dict[j[0,0]][-1]
                y_min = y_min/task_color_dict[j[0,0]][-1]
                y_max = np.max([y_max, y_nodr[0]])
                y_min = np.min([y_min, y_nodr[0]])
                

                y = np.around(y, decimals=3).tolist() 
                y_std = np.around(y_std, decimals=3).tolist() 
                y_up = np.around(np.array(y) + np.array(y_std), 3).tolist() 
                y_down = np.around(np.array(y) - np.array(y_std), 3).tolist() 
                y_nodr = np.around(y_nodr, decimals=3).tolist()
                y_nodr_std = np.around(y_nodr_std, decimals=3).tolist()
                y_ci = np.around(y_ci, decimals=3).tolist()
                y_max = np.around(y_max, decimals=3)
                y_min = np.around(y_min, decimals=3)
                print (y, y_up, y_down)
                print (y_max, y_min)

                '''
                xnew = np.linspace(ks[0], ks[-1], 50)  
                spl = make_interp_spline(ks, y, k=3) 
                ynew = spl(xnew)
                y_intersect = np.interp(ks, xnew, ynew)
                print (f"ynew: {ynew}, {len(ynew)}")
                print (f"yintersect: {y_intersect}")
                '''
                
                
                label = task_grps_label[grp_no]
                #ax.errorbar(ks, y, y_std)
                #ax.fill_between(ks, [y_nodr[0] - y_nodr_std[0]]*len(ks), [y_nodr[0] + y_nodr_std[0]]*len(ks), alpha=0.2, interpolate=False, color="yellow")
                ax.fill_between(ks, y_down, y_up, alpha=0.3, interpolate=False, color="gray")                
                #ax.plot(ks, y_ci*len(ks), label=f"{label} 95% CI L", \
                        #color=task_color_dict[j[0]][2],\
                #        marker="", markersize=14, linestyle="-.", linewidth=6)
                ax.plot(ks, y_nodr*len(ks), label=f"{label} no dr", \
                        #color=task_color_dict[j[0]][1], \
                        marker="", markersize=14, linestyle="dotted", linewidth=10)
                ax.plot(ks, y, label=f"{label}", \
                        #color=task_color_dict[j[0]][0],\
                        marker="^", markersize=30, linestyle="-", linewidth=4)

                ax.set_xscale('log')
                ax.set_xticks(ks)
                ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
                #ax.legend(fontsize=34, frameon=False, loc=0, markerfirst=False)
                ax.set_title(f"{task_grps_name[grp_no]}; N = {i}", fontsize=60)
                ax.set_xlabel("K", fontsize=60)
                ax.set_ylabel(f"{task_grps_score[grp_no]}", fontsize=60)
                if not (np.isnan(y_min) or np.isnan(y_max)): ax.set_ylim(y_min, y_max)
                ax.tick_params(axis='both', which='major', labelsize=60)
                ax.grid(axis='x')
                fig.savefig(f"./formatted_results/new_graphs_svg/N_{i}_{task_grps_file[grp_no]}.svg", \
                                bbox_inches='tight', pad_inches=0.5, format='svg', dpi=1200)
                fig.savefig(f"./formatted_results/new_graphs/N_{i}_{task_grps_file[grp_no]}.png", \
                                bbox_inches="tight", pad_inches=0.5)
                fig.clf()
                print (f"Saved: N_{i}_{task_grps_file[grp_no]}.png")
                #break
        #break
        
                
        





