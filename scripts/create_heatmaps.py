import numpy as np
import matplotlib.pyplot as plt
from help_function import read_json
import seaborn as sns
import pandas as pd
from scipy.stats import kruskal


EXPERIMENT_NAME = "Exp1"
DATA_PATH = "exp_data/export"

def create_sns_heatmap(x, y, ax, color = None):
    d = {"x": x, "y": y}
    df = pd.DataFrame(data=d)
    sns.kdeplot(df, x="x", y="y", fill=True, alpha=0.75, cbar=True, thresh=.1, color=color)

    im = plt.imread('./data/scenario.png')
    # Coordinates taken from unity
    ax.imshow(im, extent=[-9.25, 23.34, -11.2, 11.3], aspect='equal')

    return ax

def create_heatmap_from_data(data : dict, scene_identifier = None, person = None, color = None):
    x = []
    y = []
    for index, simulation in data.items():
        loginTime = ""
        for player in simulation["players"].values():
            if player["roleLogIn"] > loginTime:
                loginTime = player["roleLogIn"]
        for player in simulation["players"].values():
            if scene_identifier and scene_identifier != player["sceneName"]:
                continue
            if person and person != player["role"]:
                continue
            count = 0
            for body_item in player["Body"]:
                if body_item["serverTime"] < loginTime:
                    continue
                count += 1

                pos = body_item["position"]

                y.append(pos["x"])
                x.append(pos["z"])

    return create_sns_heatmap(x, y, fig, color), (x,y)


def check_if_simulation_is_restricted(simulation_data):
    for sim in sim_without_restrictions:
        for player in simulation_data[sim]["players"].keys():
            simulation_data[sim]["players"][player]["sceneName"] = EXPERIMENT_NAME + "_NO"


def calculate_wallis(rest_x, rest_y, clean_x, clean_y):
    rest_dist = [np.sqrt(x**2 + y**2) for x, y in zip(rest_x, rest_y)]
    clean_dist = [np.sqrt(x**2 + y**2) for x, y in zip(clean_x, clean_y)]
    stat, p_value = kruskal(rest_dist, clean_dist)

    print("Kruskal-Wallis Test Statistic:", stat)
    print("p-value:", p_value)



sim_without_restrictions = ["1", "5", "8", "13", "14"]
if __name__ == '__main__':
    data = read_json(f"{DATA_PATH}/exported_experiments_merged_distance.json.gz", True)
    check_if_simulation_is_restricted(data)

    colors = sns.color_palette(n_colors=12).as_hex()

    fig = plt.figure(figsize=(5, 4))
    plt.axis("off")
    ax1 = fig.add_subplot()
    ax1.set_title("Vision Restriction")
    ax1.set_xlabel("y")
    ax1.set_ylabel("x")
    plt, (p1_x, p1_y) = create_heatmap_from_data(data, EXPERIMENT_NAME, "Person 1")
    plt.savefig("results/heatmap_vision.png", dpi=150, transparent=True)


    fig = plt.figure(figsize=(5, 4))
    ax2 = fig.add_subplot()
    ax2.set_title("Audio Restriction")
    ax2.set_xlabel("y")
    ax2.set_ylabel("x")
    plt, (p2_x, p2_y) = create_heatmap_from_data(data, EXPERIMENT_NAME, "Person 2") #, color=colors[2])
    plt.savefig("results/heatmap_audio.png", dpi=150, transparent=True)

    fig = plt.figure(figsize=(5, 4))
    ax3 = fig.add_subplot()
    ax3.set_title("Interaction Restriction")
    ax3.set_xlabel("y")
    ax3.set_ylabel("x")
    plt, (p3_x, p3_y) = create_heatmap_from_data(data, EXPERIMENT_NAME, "Person 3") #, color=colors[3])
    plt.savefig("results/heatmap_interaction.png", dpi=150, transparent=True)

    fig = plt.figure(figsize=(5, 4))
    ax4 = fig.add_subplot()
    ax4.set_title("No Restrictions")
    ax4.set_xlabel("y")
    ax4.set_ylabel("x")
    plt, (p0_x, p0_y) = create_heatmap_from_data(data, EXPERIMENT_NAME + "_NO", color=colors[0])
    plt.savefig("results/heatmap_no_restriction.png", dpi=150, transparent=True)


    print("Wallis Test NoRestriction - Person 1")
    calculate_wallis(p1_x, p1_y, p0_x, p0_y)
    print("Wallis Test NoRestriction -  Person 2")
    calculate_wallis(p2_x, p2_y, p0_x, p0_y)
    print("Wallis Test NoRestriction -  Person 3")
    calculate_wallis(p3_x, p3_y, p0_x, p0_y)

    print("Wallis Test Person 1 - Person 2")
    calculate_wallis(p1_x, p1_y, p2_x, p2_y)
    print("Wallis Test Person 1 - Person 3")
    calculate_wallis(p1_x, p1_y, p3_x, p3_y)

    print("Wallis Test Person 2 - Person 3")
    calculate_wallis(p2_x, p2_y, p3_x, p3_y)
