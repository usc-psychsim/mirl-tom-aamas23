import plotly
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from model_learning.util.io import create_clear_dir
from model_learning.util.plot_new import *

dummy_plotly()  # ALWAYS DO THIS BEFORE A BATCH OF PLOTTING !!!

IMG_EXT = 'pdf'
OUTPUT_DIR = os.path.join('output/examples', 'timeseries_test')

create_clear_dir(OUTPUT_DIR, clear=True)

# ========================================================
num_steps = 300
data = np.random.uniform(-1, 1, num_steps)
data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.cumsum(data).reshape(-1, 1)).flatten()
data = {'random': data}

file_name = 'one_line_mean'
plot_timeseries(data, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, show_plot=False, show_legend=False)

# ========================================================
num_steps = 300
data = OrderedDict({
    'linear': 2 * np.arange(num_steps) / num_steps - 1,
    'inv linear': -2 * np.arange(num_steps) / num_steps + 1,
    'random': data['random'],
    'sine': np.sin(np.arange(num_steps) / (2 * np.pi)),
    'cosine': np.cos(np.arange(num_steps) / (2 * np.pi))
})

file_name = 'multiple_equal_length'
plot_timeseries(data, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, show_plot=False)

# ========================================================
num_steps = np.random.uniform(num_steps - 100, num_steps + 100, 4)
data = OrderedDict({
    'linear': 2 * np.arange(num_steps[0]) / num_steps[0] - 1,
    'inv linear': -2 * np.arange(num_steps[1]) / num_steps[1] + 1,
    'random': data['random'],
    'sine': np.sin(np.arange(num_steps[2]) / (2 * np.pi)),
    'cosine': np.cos(np.arange(num_steps[3]) / (2 * np.pi))
})

file_name = 'multiple_diff_length'
plot_timeseries(data, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, show_plot=False,
                palette=None, template='plotly_dark')

# ========================================================

file_name = 'multi_diff_len_average'
plot_timeseries(data, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, show_plot=False, average=True)

# ========================================================

file_name = 'multi_diff_len_norm'
plot_timeseries(data, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, show_plot=False, plot_mean=True, normalize_samples=100)

# ========================================================

num_episodes = 10
dfs = []
for e in range(num_episodes):
    num_steps = np.random.randint(200, 400)
    data = np.random.uniform(-1, 1, num_steps)
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.cumsum(data).reshape(-1, 1)).flatten()
    df = pd.DataFrame({'Dimension': data})
    df['Episode'] = e
    dfs.append(df)
df = pd.concat(dfs)

file_name = 'var_length_group'
plot_timeseries(df, 'Some Data',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Dimensions', y_min=-1, y_max=1, plot_mean=True, group_by='Episode',
                palette='Spectral')

# ========================================================

num_episodes = 100
dfs = []
for e in range(num_episodes):
    num_steps = np.random.randint(200, 400)
    data = np.random.uniform(-1, 1, num_steps)
    data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(np.cumsum(data).reshape(-1, 1)).flatten()
    df = pd.DataFrame({'Dimension': data})
    df['Episode'] = e
    dfs.append(df)
df = pd.concat(dfs)

file_name = 'var_length_group_average'
plot_timeseries(df, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, group_by='Episode', average=True)

# ========================================================

num_episodes = 100
dfs = []
for e in range(num_episodes):
    num_steps = np.random.randint(200, 400, 2)
    df = pd.DataFrame.from_dict({'Sine': np.sin((np.arange(num_steps[0]) + e) / (2 * np.pi)),
                                 'Cosine': np.cos((np.arange(num_steps[1]) + e) / (2 * np.pi))},
                                orient='index').transpose()
    df['Episode'] = e
    dfs.append(df)
df = pd.concat(dfs)

file_name = 'var_length_multi_group_avg'
plot_timeseries(df, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, group_by='Episode', average=True)

# ========================================================

file_name = 'var_length_multi_group_avg_norm'
plot_timeseries(df, 'Some Functions',
                output_img=os.path.join(OUTPUT_DIR, f'{file_name}.{IMG_EXT}'), x_label='Steps', y_label='Score',
                var_label='Functions', y_min=-1, y_max=1, plot_mean=True, group_by='Episode', average=True,
                normalize_samples=500)

fig = plotly.io.read_json(os.path.join(OUTPUT_DIR, f'{file_name}.json'))
fig.show()
