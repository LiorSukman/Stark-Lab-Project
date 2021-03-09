This folder contains several utility functions

Files:
- get_dirs.py: Generates a list of the names of session recordings, based on the CellTypeClassification.mat data

- VIS_features.py: Contains code that generates the visualizations that were used in the project's paper. These include density distributions
	command line arguments:
		* --graph_type = visualization type (can be bar, hist or mat)', default = 'bar')
		* --data_path = path to data
		* --num_fets = number of features in the data (relevant for the bar graph)
		* --index = feature index (relevant for the hist graph)
		* --bins_start = start of bins (relevant for the hist graph)
		* --bins_end = end of bins (relevant for the hist graph)
		* --bins_num = number of bins (relevant for the hist graph)
		* --title = graph title (relevant for the hist graph)
		* --x_label = x axis label (relevant for the hist graph)
		* --y_label = y axis label (relevant for the hist graph)
