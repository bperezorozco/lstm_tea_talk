import scipy.io
import numpy as np

#frames the signal @x into frames of size @frame_size separated by @hop samples
def get_frames( x, frame_size, hop ):
    start_idx = 0
    end_idx = frame_size
    frames = []
    limit = x.shape[1]
    
    while end_idx <= limit:
        frames.append( x[:, start_idx:end_idx] )
        start_idx = start_idx + hop
        end_idx = start_idx + frame_size
    
    frames = np.float32(frames)
    
    return np.swapaxes( frames, 1, 2 )

def long_prediction( model, x, length, timesteps ):
    seed = x
    for i in range( length ):
        y = model.predict( np.expand_dims( seed, axis= 0 ) )
        x = np.vstack( (x, y[0, -1, 0]) )
        seed = x[-timesteps:]
    
    return x

def prepare_data( data ):
    sample_length = 1
    timesteps = 99
    hop = 1

    data = ( data - data[0].mean() ) / data[0].std()

    X_train = get_frames( data[:, 0:-1], timesteps, hop )
    y_train = get_frames( data[:, 1:], timesteps, hop )

    return ( X_train, y_train )


