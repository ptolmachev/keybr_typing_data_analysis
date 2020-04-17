import pickle
from matplotlib import pyplot as plt
from src.load_data import preprocess_data
from src.plots import plot_char_stats, plot_speed

if __name__ == '__main__':
    name = 'typing-data'
    folder_name = f'../data/{name}.json'
    data = preprocess_data(folder_name)
    pickle.dump(data, open(f'../data/{name}.pkl', 'wb+'))
    # data = pickle.load(open(f"../data/{name}.pkl", 'rb+'))
    fig_char_stats = plot_char_stats(data, window = 100)
    fig_char_stats.savefig("../img/char_stats.png")
    plt.close()
    speed_data = plot_speed(data)
    speed_data.savefig("../img/speed_stats.png")
    plt.close()