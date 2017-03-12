data_dir = "../_DATA/CarND/p3_behavioral_cloning/set_000/"
image_dir = "IMG/"
driving_data_csv = "driving_log.csv"
batch_size = 32 #256
nb_epoch = 3 

should_retrain_existing_model = False
saved_model = "model_epoch_33_val_acc_0.0.h5"
previous_trained_epochs = 30