def remove_zero_angles(sample_set: list):
    
    import numpy as np
    from DataHelper import plot_histogram, get_steering_values, find_nearest
    
    print("len(sample_set)", len(sample_set))
    indexes_to_keep = []
    
    steering_angles = get_steering_values(sample_set)
    plot_histogram("steering values", steering_angles, change_step=0.01)

    for index in range (len(steering_angles)):
        angle = steering_angles[index]
        if angle != 0: 
            indexes_to_keep.append(index)

    print("len(indexes_to_keep)", len(indexes_to_keep))

    set_to_keep = []
    for index in indexes_to_keep:
        set_to_keep.append(sample_set[index])
 
    # release the memory
    training_to_keep = []
    indexes_to_keep = []

    print("len(set_to_keep)", len(set_to_keep))

    steering_angles = get_steering_values(set_to_keep)
    plot_histogram("steering values", steering_angles, change_step=0.01)
    return set_to_keep, steering_angles
