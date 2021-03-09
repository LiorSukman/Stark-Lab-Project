This folder conatins all of the machine learning part of the project. Several supervised learning methods were used in this project (SVM, RF, NN) and several unsupervised ones (GMM).

Files:
- data.txt: contains paths to different datasets (each chunk size configuration gets its own folder)

- dataset_creator.py: creates the different datasets (train, dev, test)
    command line arguments:
        * --per_train = percentage of the data to be in train set
        * --per_dev = percentage of the data to be in developemnt set
        * --per_test = percentage of the data to be in test set
        * --datasets = path to data dirs
        * --should_filter = filter unlabeled units out
        * --verbos = print information about datasets
        * --save_path = path to save datasets, make sure path exists
        * --keep = indices to keep, make sure to put -1 in there for the label

- feature_dropping_svm.py: Responsible for performing feature dropping and assesment of different combinations of features.
    command line arguments:
        * --dataset_path = path to the dataset, assume it was created

- gs_gmm.py: Performs a grid search for the GMM unsupervised learning method
    command line arguments:
        * --use_tsne = use tSNE for visualization
        * --tsne_n_components = number of tSNE components
        * --use_scale = scale the data
        * --use_pca = transform the data using PCA
        * --pca_n_components = number of PCA components
        * --use_ica = transform the data using ICA
        * --ica_n_components = number of ICA components
        * --dataset_path = path to the dataset, assume it was created
        * --verbos = verbosity level (bool)
        * --saving_path = path to save graphs, assumed to be created
        * --min_search_components = minimal number of gmm components to check
        * --max_search_components = maximal number of gmm components to check

- gs_rf.py: Performs a grid search for the RF supervised learning method
	command line arguments:
		* --dataset_path = path to the dataset, assume it was created
		* --verbos = verbosity level (bool)
		* --n_estimators_min = minimal power of n_estimators (base 10)
		* --n_estimators_max = maximal power of n_estimators (base 10)
		* --n_estimators_num = number of n_estimators values
		* --max_depth_min = minimal power of max_depth (base 10)
		* --max_depth_max = maximal power of max_depth (base 10)
		* --max_depth_num = number of max_depth values
		* --min_samples_splits_min = minimal power of min_samples_splits (base 2)
		* --min_samples_splits_max = maximal power of min_samples_splits (base 2)
		* --min_samples_splits_num = number of min_samples_splits values
		* --min_samples_leafs_min = minimal power of min_samples_leafs (base 2)
		* --min_samples_leafs_max = maximal power of min_samples_leafs (base 2)
		* --min_samples_leafs_num = number of min_samples_leafs values
		* --n = number of repetitions
	
- gs_svm.py: Performs a grid search for the SVM supervised learning method 
	command line arguments:
		* --dataset_path = path to the dataset, assume it was created
		* --verbos type=bool = verbosity level (bool)
		* --saving_path = path to save graphs, assumed to be created
		* --min_gamma  = minimal power of gamma (base 10)
		* --max_gamma  = maximal power of gamma (base 10)
		* --num_gamma  = number of gamma values
		* --min_c  = minimal power of C (base 10)
		* --max_c  = maximal power of C (base 10)
		* --num_c = number of C values
		* --kernel = kernael for SVM (notice that different kernels than rbd might require more parameters)

- NN_evaluator.py: Evaluates a specific neural network model

- NN_model.py: Contains the actual neural network model

- NN_param_search.py: Parameter search for the NN
    command line arguments:
        * --epochs = number of epochs (times to go over the data)
        * --patience = number of epochs to tolerate with no improvement on the dev set
        * --batch_size = number of examples in a batch
        * --learning_rate = learning rate for the optimizer
        * --optimizer = optimizer to use, can be sgd, adam or adadelta
        * --n1 = size of the first hidden layer
        * --n2 = size of the second hidden layer
        * --f1 = activation function before first hidden layer
        * --f2 = activation function before second hidden layer
        * --classes = size of the output layer (number of classes)
        * --features = size of the input layer (number of features)
        * --dataset_path = path to the dataset, assume it was created
        * --loading_path = path to a trained model to evaluate
        * --saving_path = path to save models while training, assumes path exists
		* --verbos = verbosity level (bool)
		* --ns_min = minimal power for hidden layers size (base 2)
		* --ns_max = maximal power for hidden layers size (base 2)
		* --ns_num = number of hidden layer sizes to check
		* --default_lr = default learning rate for phase 1
		* --default_opt = default optimizer for phase 1
		* --lrs_min = minimal power for learning rate (base 10)
		* --lrs_max = maximal power for learning rate (base 10)
		* --lrs_num = number of learning rates to check
		* --n = number of repetitions


- NN_pipeline.py: Performs the entire learning and evaluation processes for the neural network model.
    command line arguments:
        * --epochs = number of epochs (times to go over the data)
        * --patience = number of epochs to tolerate with no improvement on the dev set
        * --batch_size = number of examples in a batch
        * --learning_rate = learning rate for the optimizer
        * --optimizer = optimizer to use, can be sgd, adam or adadelta
        * --n1 = size of the first hidden layer
        * --n2 = size of the second hidden layer
        * --f1 = activation function before first hidden layer
        * --f2 = activation function before second hidden layer
        * --classes = size of the output layer (number of classes)
        * --features = size of the input layer (number of features)
        * --dataset_path = path to the dataset, assume it was created
        * --loading_path = path to a trained model to evaluate
        * --saving_path = path to save models while training, assumes path exists

- NN_trainer.py: Responsible for the training of the neural network model

- ML_util.py: Contains several utility functions for the folder

- SVM_RF.py: Runs the SVM and RF models and presents their results
	command line arguments:
		* --model = Model to train, now supporting svm and rf
		* --dataset_path = path to the dataset, assume it was created
		* --verbos = verbosity level (bool)
		* --saving_path = path to save models, assumed to be created
		* --loading_path = path to load models from, assumed to be created and contain the models
		* --gamma = gamma value for SVM model
		* --C = C value for SVM model
		* --kernel = kernael for SVM (notice that different kernels than rbf might require more parameters)
		* --use_scale = apply scaling on the data
		* --use_pca = apply PCA on the data', default = False)
		* --pca_n_components = number of PCA componenets
		* --use_ica = apply ICA on the data
		* --ica_n_components = number of ICA componenets
		* --visualize = visualize the model
		* --n_estimators = n_estimators value for RF
		* --max_dept = max_depth value for RF
		* --min_samples_split = min_samples_split value for RF
		* --min_samples_leaf = min_samples_leaf value for RF


- VIS_heatmap.py: Creates a visual heatmap for grid searches

- VIS_model.py: Performs a visualization of different supervised models