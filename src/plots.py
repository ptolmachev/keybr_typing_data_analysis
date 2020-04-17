import operator

from matplotlib import pyplot as plt
import numpy as np
import pickle
from numpy.polynomial.polynomial import Polynomial

def plot_speed(data):
    speed = (1.0/5.0) *  data['length'] / (data['time']/1000/60)
    x = np.arange(len(speed))
    y = speed
    fig, ax = plt.subplots(2,1, figsize = (20,10)) #plt.figure(figsize = (20,10))
    # ax[0].set_title("Words per minute", fontsize=24)
    fit = Polynomial.fit(x, y, deg=12)
    ax[0].scatter(x, y, linewidth=2, color = 'limegreen', label = 'speed (raw data)')
    ax[0].plot(x, fit(np.array(x).astype(float)), linewidth=4, color = 'forestgreen', label = 'speed (polynomial fit)')
    ax[0].legend(fontsize=24)
    # ax[0].set_xlabel("sample num", fontsize=24)
    # ax[0].set_xlabel()
    ax[0].set_xticklabels([])

    ax[0].set_ylabel("Speed, words per min", fontsize=24)
    ax[0].set_ylim([0, 1.1 * np.max(y)])
    ax[0].grid(True)

    errors = data['errors']
    x = np.arange(len(errors))
    y = errors
    fit = Polynomial.fit(x, y, deg=12)
    ax[1].scatter(x, y, s=2, color = 'red', label = 'errors (raw data)')
    ax[1].plot(x, fit(np.array(x).astype(float)), linewidth=4, color = 'crimson', label = 'errors (polynomial fit)')
    ax[1].legend(fontsize=24)
    ax[1].set_xlabel("sample num", fontsize=24)
    ax[1].set_ylabel("Errors", fontsize=24)
    ax[1].set_ylim([0, 1.1 * np.max(y)])
    ax[1].grid(True)
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.tight_layout()
    plt.show(block = True)
    return fig

def plot_char_stats(data, window):
    char_codes = [32, 101, 105, 108, 110, 114,
                  116, 115, 97, 117, 111, 100,
                  121, 99, 104, 103, 109, 112,
                  98, 107, 118, 119, 102, 122,
                  120, 113, 106]
    chars_lbls = [str(f'\'{chr(c)}\'') for c in char_codes]
    chars = [chr(c) for c in char_codes]

    char_speeds = []
    char_speeds_std = []
    char_times = []
    char_times_std = []
    char_hits = []
    char_hits_std = []
    char_miss = []
    char_miss_std = []
    char_mrs = []
    char_mrs_std = []
    for char in chars:
        timeToType = data[f'\'{char}\'_timeToType'] / 1000 #in seconds
        speed = 60 / timeToType / 5.0 # in wpm
        speed = np.array(speed.dropna())
        h = np.array(data[f'\'{char}\'_hitCount'].dropna())
        m = np.array(data[f'\'{char}\'_missCount'].dropna())
        miss_ratio = 100 * m / (m + h)

        if len(speed) < window:
            char_speeds.append(np.nanmean(speed))
            char_speeds_std.append(np.nanstd(speed))
            char_mrs.append(np.nanmean(miss_ratio))
            char_mrs_std.append(np.nanstd(miss_ratio))
            char_times.append(np.nanmean(timeToType))
            char_times_std.append(np.nanstd(timeToType))
            char_hits.append(np.nanmean(h))
            char_hits_std.append(np.nanstd(h))
            char_miss.append(np.nanmean(m))
            char_miss_std.append(np.nanstd(m))
        else:
            char_speeds.append(np.nanmean(speed[:-window]))
            char_speeds_std.append(np.nanstd(speed[:-window]))
            char_mrs.append(np.nanmean(miss_ratio[:-window]))
            char_mrs_std.append(np.nanstd(miss_ratio[:-window]))
            char_times.append(np.nanmean(timeToType[:-window]))
            char_times_std.append(np.nanstd(timeToType[:-window]))
            char_hits.append(np.nanmean(h[:-window]))
            char_hits_std.append(np.nanstd(h[:-window]))
            char_miss.append(np.nanmean(m[:-window]))
            char_miss_std.append(np.nanstd(m[:-window]))

    fig, ax = plt.subplots(4, 1, figsize = (20,10))
    y, x, z = zip(*sorted(zip(char_times, np.arange(len(chars)), char_times_std)))
    y = y[::-1]
    x = x[::-1]
    z = z[::-1]
    ax[0].bar(np.arange(len(y)), y,  yerr=z, align='center', alpha=0.5, ecolor='black',
              capsize=10, label='Time to type, sec')
    ax[0].set_xticks(np.arange(0, len(chars) + 1, 1.0))
    ax[0].set_xticklabels(list(map(chars_lbls.__getitem__, x)), fontsize = 16)
    ax[0].set_ylabel('Time to type, sec', fontsize = 16)

    y, x, z = zip(*sorted(zip(char_mrs, np.arange(len(chars)), char_mrs_std)))
    ax[3].bar(np.arange(len(y)), (np.array(y)).astype(int),  yerr=np.array(z), color = 'violet', align='center', alpha=0.5, ecolor='black',
              capsize=10, label='Miss ratio')
    ax[3].set_xticks(np.arange(0, len(chars) + 1, 1.0))
    ax[3].set_xticklabels(list(map(chars_lbls.__getitem__, x)), fontsize = 16)
    ax[3].set_ylabel('Miss ratio, %', fontsize = 16)

    _, y, x, z = zip(*sorted(zip(char_mrs, char_hits, np.arange(len(chars)), char_hits_std)))
    ax[1].bar(np.arange(len(y)), y,  yerr=z, color = 'green', align='center', alpha=0.5, ecolor='black',
              capsize=10, label='Num hits')
    ax[1].set_xticks(np.arange(0, len(chars) + 1, 1.0))
    ax[1].set_xticklabels(list(map(chars_lbls.__getitem__, x)), fontsize = 16)
    ax[1].set_ylabel('Num hits', fontsize = 16)

    _, y, x, z = zip(*sorted(zip(char_mrs, char_miss, np.arange(len(chars)), char_miss_std)))
    ax[2].bar(np.arange(len(y)), y,  yerr=z, color = 'red', align='center', alpha=0.5, ecolor='black',
              capsize=10, label='Num misses')
    ax[2].set_xticks(np.arange(0, len(chars) + 1, 1.0))
    ax[2].set_xticklabels(list(map(chars_lbls.__getitem__, x)), fontsize = 16)
    ax[2].set_ylabel('Num misses', fontsize = 16)

    for i in range(len(ax)):
        ax[i].grid(True)
        ax[i].legend(loc=1)
        ax[i].set_ylim([0, None])
    plt.tight_layout()
    plt.subplots_adjust(wspace=None, hspace=None)
    plt.show(block = True)
    return fig

