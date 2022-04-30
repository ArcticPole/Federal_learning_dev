# Federal_learning_dev

Federal learning 3.0
created by Xiao Peng, Lin Ze, Hao Yin, Dr. Wang Chuang

There are 3 stage in this projection.
first_stage is for console to console transfer learning
	s1_send.py is for simulating sending pkl from console A
	s1_recieve is function program for receiving pkl
	model_feature.py is function program for catch weight feature in pkl
	model_feature_another.py is function program for catch weight feature in another pkl
	xiao_fault_merge.py is function program for merge two information from two console
	State_one_merge_example.py is for simulate receiving pkl from console A, merge a new model and checking its accuracy
	/merge_models is store pkl after merging
example file: s1_send.py and State_one_merge_example.py

second_stage is for develop cloud model and develop personalized model in console
	function.py is function program for sevaral function used in stage 2
	hy_local is for simulating receiving model for cloud and personalize model in console
	s2_cloud is for simulating develop global model and sending to console
	s2_local is function program for receiving model
	xiao_global_model.py is function program for develop cloud model
	/cloud_para is to store cloud model
	/local_models is to store local model
example file: s2_cloud, hy_local

third_stage is for sending personlized information to cloud and cloud produce personalized model for console, when console have a new working scene
	s3_cloud_example.py is for simulating cloud process
	s3_local_example.py is for simulating local(console) process
	s3_recieve_cloud.py is function program for cloud receiving information from console
	s3_send_cloud.py is function program for cloud sending information to console
	s3_send_recieve_local is function program for local process
	test_samp.pth is temporary data file
	Willian_dataset_random.py is function program for producing example data with specified rate
	Willian_model_test.py is function program for checking example data's rate
	xiao_individualized_model is function program for produce personalized model by given information
	/local_data is cloud store data from local （console)
	/local_model is console store model receive from cloud
example file: s3_local_example.py, s3_cloud_example.py

tools is some public function program which every stage will use
	model.py is store several model structure may used in all stage
	xiao_dataset_random.py is to produce dataset from original csv file
	xiao_fault diagnosis.py is to produce original model
	xiao_feature_enhance.py is a algorithm to enhance feature in model's convolution kernel
	xiao_feature_enhance_max.py is another algorithm to enhance feature in model's convolution kernel
	xiao_global_feature.py is algorithm to extract weights from several pkl

/trained_model is to store some pkl we have trained
/don't use now is to store some algorithm or function file that we used previously
/lossfig is to store loss picture that produce when training model
/data is to store original data which is in csv file
/global_models is to store source model and produced model in cloud

Stage I scenario: different robots encounter different errors in the same scenario, and learn the errors that have not occurred in advance through federated learning.
Stage II scenario: the models in different scenarios form a cloud integrated model. The edge end can obtain the integrated model from the cloud, which contains the knowledge required by the edge end and can be personalized through local re learning.
Stage III scenario: models of different scenarios are saved in the cloud. Building a new scene at the edge has the error of being similar to the cloud scene. By comparing the similarity, assign the weight of each scene model, and personalize the model to the edge.
 

**************************************************************************************************
Federal learning 2.1
created by Xiao Peng, Lin Ze, Hao Yin, Dr. Wang Chuang

fault diagnosis.py is the main processing file for "overall model"
xiao_fault diagnosis.py is the main processing file for "vote model"

model.py define several CNN model, 
	which "CNN2d_classifier_xiao" is used for clip size=2304;
	"CNN2d_classifier" is used for clip size=4096 (only for previous data precessing methods);
	"CNN2d_fitting_xiao" is used for clip size=576;
	other model is not used yet.

xiao_estimate.py is used for vote model estimate, it load each part-model and test accuracy of vote model.

xiaodataset.py is for data process, compare to the previous methods, it can increase the utilization rate of data.
xiao_dataset_random.py is for data process too, this can divide the data into train and test randomly.


