README
Data folder contain the matlab files 
Save models folder contain the results tables and models.
First we need to load the data using importData file - the processed data has been processed in MATLAB. the function load_and_scale_data() import the MATLAB files using get_data() and create the data structure we need and scale it. 

note that for competaility the user MUST change the paths in lines 17-21 to the appropriate path to the 'data for python' directory.

Run SVM:
In this part we will run svm with diff Dimensions.
for 2 Dimensions we run the func: find_best_features(X,y, features_names)
for 3 Dimensions we run the func: find_best_features_trip(X, y, features_names)
for 4 Dimensions we run the func: find_best_features_four(X,y, features_names,four_list)
for 5 Dimensions we run the func: find_best_features_five(X,y, features_names,five_list)

To create the four_list use the func get_four_list(X) after you run the 2 dimmensions beacuse the func use the retult to calculate the Four_list.
The same idea to with get_five_list(X, score_list). Run 3 dimmensions SVM before.
To print the SVM use new_plot_2(C, gamma, X, X_tag, y_tag, pair, features_names).
Plot the result of the SVMs:
in file resultSVM we will load all the result with load_final_data() and plot the graph:
 
With the func run_final_svm(X,y,i ,j, features_names) you can plot any 2 dim SVM couple (i,j) as save in the list_array2 you want.
Run GMM:
The function get_GMM_model(X, n_comp, best_type, n_init, true_labels, join_labels) return the GMM model with the error as calc by the func calc_error(true_labels, model_labels). 

the funcfion learnig_GMM_BIC(X) - run GMM with 4 difference covariance_type: 'spherical', 'tied', 'diag', 'full' and calculate the BIC for each model. (We didnt use it - only for learning how to use GMM)
We use the function run_BIC_GMM(X) - create plot of numbers of GMM model with difference components numbers vs. BIC score. The model can be set in the function - user can change the model (covariance type for example).
the function plot_GMM_2d(X, n_comp, best_type, n_init, true_labels, ax) - plot the required GMM model only for 2 dimmensions data structure (we use it only for the TSNE part).
for saveing a model use : joblib.dump(clf, 'model_name.pkl')
for load a model: clf_load = joblib.load('model_name.pkl')
Run the forest and TSNE:
This part was only for visualization of the data and to check if it can be divided to groups. Part of the code from Guy Goren project. The main idea was to use random forest algorithm for calculate destination matrix and than using it as input to TSNE - down the dimension to 2. 
For those 2 vector we run GMM and BIC to try to clustering it to group.
link to Guy Goren project: https://github.com/gorenguy/uRF_SDSS















 



 