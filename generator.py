def build_image_paths(data_dir: str, image_names: list):
    image_paths = []
    for image_name in image_names:
        image_paths.extend([data_dir + image_name])
    return image_paths

def read_images(image_paths):
    import numpy as np 
    from ImageHelper import read_image_array, read_image_binary
    #training_features = [read_image_array(path) for path in image_paths]

    image_list = []
    for path in image_paths:
        image = np.array(read_image_binary(path))
        image_list.append(image)
    training_features = np.array(image_list) # numpy array, not just a list
    return training_features


import numpy as np


def generator(name, example_set: np.ndarray, data_dir: str, batch_size: int=32 ):
    """
    Yields batches of training or testing data every time the generator is called.
    I will use Keras to pre-process images (trim, resize)
    """
    
    import sklearn
    from DataHelper import get_image_center_values, get_steering_values
 
    yield_number = 0
    total_samples = len(example_set)
    sample_batch = []
    features = []

    while True:
        for offset in range(0, total_samples, batch_size):

            offset_end = offset + batch_size
            if offset_end > total_samples:
                offset_end = total_samples

            sample_batch = example_set[offset:offset_end]

            #print(name,
            #    "batch:", yield_number, 
            #    "size:", len(sample_batch),
            #    "index:", offset, "to", offset_end,
            #    "of", total_samples
            #    )

            labels = get_steering_values(sample_batch)
            image_names = get_image_center_values(sample_batch)
            image_paths = build_image_paths(data_dir, image_names)
            features = read_images(image_paths)
            if len(features) != len(labels):
                print("ERROR: ", len(features), " features and ", len(labels), "labels count not matching!")
            yield_number = yield_number + 1
            #sklearn.utils.shuffle( ) # I prefer not to mix them
            yield np.array(features), np.array(labels)