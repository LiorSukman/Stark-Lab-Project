# stark-project
Authors:
- Lior sukman (lior.sukman@gmail.com)
- Gil davidovic (gilnd99@gmail.com)

This project contains several directories:
- \Data: a directory containing the raw data, it is not submmited due to its large size
- \features: Contains code that computes the features used for classification
- \ml: Contains all of the code that pertains to the machine learning side of the project (both supervised and unsupervised)
- \utils: Contains several utility functions mainly used for graphical display of the preprocessed data

Files:
- clusters.py: contains the main clases that were used to represent unit data in our project (i.e Spike class used to represent a spike, and a Cluster class that contains several spikes).
- data_stats.py: Several functions that are used to display statistical data
- read_data.py: Responsible to read the optogenetic data and create Cluster and Spikes as mentioned above.
    Important - this functionality depends on the existence of a file called dirs.txt that contains the locations
    of all directories that contain that type of data. each of these directories should contain at leat one .clu file and one .spk file.
	It also depends on the existence of the file "CelltypeClassification.mat" under the "\Data" folder.
- dirs.txt: Used to indicate the location of the optogenetic data as mentioned in the read_data.py section.
	Each row in the file should contain the location of one session recording. for example: "PATH_ON_COMPUTER\es25nov11_13".
	it is possible to add numbers corresponding to shanks after each directory's path to skip the shank.
- pipeline.py: The main pipeline of the preprocessing. Responsible for calling the read_data.py file and computing the different features that reside in the features folder. 
	After computing these features for each of the units, the pipeline creates a directory with the unique name of the unit (explained inside the file)
	under the location "clustersData/UNIQUE_NAME". inside this directory is a .csv file that contains the values for each of the features for that unit.

    command line arguments:
        * --dirs_file = path to data directories file
        * --chunk_sizes = chunk sizes to create data for, can be a list
        * --save_path = path to save csv files to, make sure the directory exists
        * --spv_mat = path to SPv matrix

