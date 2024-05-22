import pickle
import numpy as np
import matplotlib.pyplot as plt

def get_percentiles(data, eval_percentages = [0, 25, 50, 75, 100]):
    percentiles = []
    for percentage in eval_percentages:
        percentiles.append(np.percentile(data, percentage))
    return percentiles

if __name__ == "__main__":
    # file = rospkg.RosPack().get_path('rl_env') + "/logging/pickles/xxx.pickle"
    picke_file = "/SemiAutonomousControl/hand_control_ws/src/hand_prosthesis_rl_control_pkgs/rl_env/logging/pickles/mia_hand_rl_PPO_20240514_172010.pickle"
    with open(picke_file, "rb") as f:
        data = pickle.load(f)
    
    for key, values in data.items():
        # Compute the sum across the fingers
        if isinstance(values[0][0], np.ndarray) or isinstance(values[0][0], list):
            values = np.sum(values, axis=2)
        
        # Compute the sum across the episodes
        values = np.array(values, dtype=np.float64)
        values_per_episode = np.sum(values, axis=1)
        print(f"Average {key}:", np.sum(values_per_episode) / values.shape[0])
        print(values_per_episode)
        # Get the percentiles
        percentiles = get_percentiles(values_per_episode)
        print(f"Percentiles {key}:", percentiles, "\n")
        
        # Plot the boxplot
        # plt.boxplot(values_per_episode)
        # plt.title(key)
        # plt.show()
        
    # Fix if the data if necessary
    # for key in data.keys():
    #     for i, sub in enumerate(data[key]):
    #         if len(sub) != 47:
    #             print(f"Index: {i}, Length: {len(sub)}")
    #             del data[key][i][:(len(data[key][i])-47)]
    # with open(picke_file, "wb") as f:
    #     data = pickle.dump(data, f)