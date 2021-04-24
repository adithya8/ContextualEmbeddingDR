import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.rcParams['svg.hashsalt'] = 42
np.random.seed(42)

#print (plt.style.available)
#https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
styles = ['seaborn-notebook', '_classic_test', 'seaborn-pastel', 'seaborn-talk', 'seaborn', 'seaborn-poster', \
        'seaborn-deep', 'seaborn-ticks', 'seaborn-paper', 'grayscale', 'seaborn-dark-palette',\
        'seaborn-whitegrid', 'classic', 'ggplot', 'seaborn-colorblind', 'seaborn-bright', 'bmh',\
        'Solarize_Light2', 'seaborn-white', 'fast', 'dark_background', 'fivethirtyeight', 'seaborn-dark', \
        'seaborn-muted', 'seaborn-darkgrid', 'tableau-colorblind10']

sns.set_style('fast')
data = pd.read_csv("./nvsk.csv")

n = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
ks = [16, 32, 64, 128, 256, 512]
task_grps = ['age', ('gen', 'gen2'), ('ext', 'ope'), ('bsag', 'sui')]
task_grps_name = ["Demographics", "Demographics", "Personality", "Mental-health"]
task_color_dict = {
    "age": ["#ff3333", "#33beff", "#33ff42", 1],
    "gen": ["#00a630", "#4e00a6", "#a60000", 1],
    "gen2": ["#ffd800","#00fffb", "#ff00a6", 1],
    "ext": ["", "", "", np.sqrt(0.70*0.77)],
    "ope": ["", "", "", np.sqrt(0.70*0.77)],
    "bsag": ["", "", "", np.sqrt(0.70*0.77)],
    "sui": ["", "", "", 1]
}

for i in n:
    for grp_no in range(len(task_grps[:-1])):
        temp_data = data[(data.N == i) & (data.Task.isin(task_grps[grp_no]))]
        fig, ax = plt.subplots(figsize=(15, 15))
        for j in temp_data.values:
            y = (j[2:8]/task_color_dict[j[0]][-1]).tolist()
            y_nodr = (j[-2:-1]/task_color_dict[j[0]][-1]).tolist()
            y_ci = (j[-1:]/task_color_dict[j[0]][-1]).tolist()

            y = np.around(y, decimals=3).tolist() 
            y_nodr = np.around(y_nodr, decimals=3).tolist()
            y_ci = np.around(y_ci, decimals=3).tolist()            

            ax.plot(ks, y_nodr*len(ks), label=f"{j[0]} no dr", \
                    #color=task_color_dict[j[0]][1], \
                    marker="", markersize=16, linestyle="dotted", linewidth=3)
            ax.plot(ks, y_ci*len(ks), label=f"{j[0]} 95% CI L", \
                    #color=task_color_dict[j[0]][2],\
                    marker="", markersize=14, linestyle="-.", linewidth=3)
            ax.plot(ks, y, label=f"{j[0]}", \
                    #color=task_color_dict[j[0]][0],\
                    marker="^", markersize=14, linestyle="-", linewidth=3)

        ax.set_xscale('log')
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.legend(fontsize=15, title="Legend")
        ax.set_title(f"{task_grps_name[grp_no]}: Scores vs K for N = {i}", fontsize=25)
        ax.set_xlabel("K", fontsize=22)
        ax.set_ylabel("Score", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True)
        fig.savefig(f"./formatted_results/graphs_svg/N_{i}_{task_grps_name[grp_no]}.svg", \
                    bbox_inches='tight', pad_inches=0.5, format='svg', dpi=1200)
        plt.clf()
        print (f"Saved: N_{i}_{task_grps_name[grp_no]}.svg")
        
