import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df_spiking = pd.read_csv("../output/spiking_data.csv", header=None)
    df_voltage = pd.read_csv("../output/voltage_data.csv", header=None)
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6), sharey=True)
    sns.heatmap(df_spiking.T, cmap="viridis", cbar_kws={'label': 'Spike (bool)'}, ax=axs[0])
    axs[0].set_xlabel("Time Step")
    axs[0].set_ylabel("Neuron Index")
    axs[0].set_title("Spiking activity")

    sns.heatmap(df_voltage.T, cmap="viridis", cbar_kws={'label': 'Voltage (mV)'}, ax=axs[1])
    axs[1].set_title("Membrane potentials")
    axs[1].set_xlabel("Time Step")

    fig.suptitle('Attractor network simulation')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
