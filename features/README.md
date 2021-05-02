This folder contains the files that are responsible for the computation of the different features. Each feature must implement two methods:

* calculate_feature(spikeList): This method gets a list of spikes and computes the relevant feature for each of them. It returns a matrix with the following dimensions: (# of spikes in the list, # of metrics the feature calaulates).

* get_headers(): This method returns the names of the metrics that the feature calculates

Files:
- FET_channel_contrst_feature.py: Computes the channel contrast feature
- FET_da.py: Computes the direction agreebleness feature
- FET_depolarization_graph.py: Computes the depolarization graph feature
- FET_geometrical_estimation.py: Computes the geometrical estimation feature
- FET_spd.py: Computes the spatial dispersion feature
- FET_time_lag.py: Computes the time lag feature

further information about the exact calculation of each feature can be found at the project's paper.