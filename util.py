import matplotlib.colors as pltcolors
import numpy as np
import os
import re
import warnings

mask_color_names = ['purple', 'blue', 'pink', 'red', 'orange', 'teal', 'magenta', 'olive', 'ecru', 'yellow', 'lilac', 'peach', 'pale green', 'sky blue', 'white', 'mustard', 
                    'grey', 'cyan', 'light brown', 'bright pink', 'ice blue', 'dark green', 'mauve', 'dark red', 'red orange', 'greyish purple', 'neon purple', 'cobalt',
                    'medium blue', 'clay', 'avocado', 'pinky red', 'orange yellow', 'ivory', 'wheat', 'shamrock green', 'pear', 'ultramarine blue', 'greeny brown',
                    'very light pink', 'carnation', 'dusty red', 'petrol',  'pumpkin orange', 'saffron', 'greenish turquoise', 'light khaki', 'bluey grey', 'hazel',
                    'topaz', 'light pea green', 'battleship grey', 'deep brown', 'bruise', 'dark cream', 'stormy blue', 'orange pink', 'candy pink', 'bland', 'macaroni and cheese', 
                    'cloudy blue', 'snot', 'auburn', 'strawberry']
MASK_COLORS = [np.array(pltcolors.to_rgb(f'xkcd:{color_name}')) * 255 for color_name in mask_color_names]
MASK_COLORS= np.array(MASK_COLORS, dtype='uint8')

def parse_train_step(ckpt_name):
    try:
        train_step = int(re.split('\D', ckpt_name.split('step=')[1], maxsplit=1)[0])
    except:
        train_step = 0
    return train_step

def get_checkpoint_path(checkpoint_dir):
    """
    Given a directory containing model checkpoints, load the one with the highest number of train steps.
    """
    checkpoint_fnames = [fname for fname in os.listdir(checkpoint_dir) if fname.endswith('.ckpt')]
    if not checkpoint_fnames:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_dir}')

    best_train_step = 0
    for fname in checkpoint_fnames:
        train_step = parse_train_step(fname)
        if train_step > best_train_step:
            best_train_step = train_step
            best_checkpoint_fname = fname

    if best_train_step == 0:
        warnings.warn('Failed to parse train step from checkpoint path(s), the most recent checkpoint may not be loaded.',
                      RuntimeWarning)
        best_checkpoint_fname = checkpoint_fnames[0]
    
    checkpoint_fname = best_checkpoint_fname
    train_step = best_train_step

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_fname)

    return (checkpoint_path, train_step)
    
def fourier_embeddings(data, num_freqs=10, max_sampling_rate=60):
    freqs = np.linspace(1, max_sampling_rate, num_freqs) * (np.pi/2)
    num_embeds = 2*num_freqs + 1
    output = np.zeros(data.shape + (num_embeds,))
    output[..., 0] = data
    for (ind, freq) in enumerate(freqs):
        output[..., 2*ind + 1] = np.sin(freq*data)
        output[..., 2*ind + 2] = np.cos(freq*data)
        
    # Flatten the last dimension
    return output.reshape(data.shape[:-1] + (-1,))