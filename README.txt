This directory contains the necessary MATLAB code to replicate simulation results in 
"Natural Gradient Hybrid Variational Inference with Application to Deep Mixed Models"
Authors: Weiben Zhang, Michael Smith, Worapree Maneesoonthorn and Rubén Loaiza-Maya

Each folder contains MATLAB scripts to replicate the results and 
a 'Service functions' subfolder with necessary functions for the script.

The results presented in the paper were produced using MATLAB R2022a on a Windows 10 Pro 64-bit operating system, 
with a 2.50 GHz 11th Gen Intel(R) Core(TM) i9-11900 processor.
Two exceptions are the scripts 'Eg2_DeepGLMM_robust.m' and 'Eg3_DeepGLMM_robust.m', which were run on a high performance computation system (Spartan).
To run these two scripts on a standard computer, we recommend changing the 'parfor' loop to a 'for' loop to avoid out-of-memory errors.

Directories:
'Example 1'	This folder contains code for the simulation and estimation of a linear mixed model.  
		'VI_Hybrid_LMM.m' 		Main script to replicate the results. This script calls for functions in the 'Service functions' subfolder.
		Main functions include:
			'data_sim.m'		Simulates data from linear mixed model.
			'LMM_MCMC.m'		Estimates linear mixed model using MCMC.
			'davi_lmm_train.m'	Estimates linear mixed model using DAVI.
			'hybrid_lmm_train.m'	Estimates linear mixed model using hybrid methods. 
						

'Example 2'	This folder contains code for the simulation and estimation of a Gaussian deep mixed model.
		'Eg2_DeepLMM_a.m' 			Script to replicate the results in example 2(a).
		'Eg2_DeepLMM_a_robust.m'	Script to replicate the boxplot in example 2(a). 
		'Eg2_DeepLMM_b.m' 			Script to replicate the results in example 2(b).
							        These scripts call for functions in the 'Service functions' subfolder.  
		Main functions include:
			'data_sim.m'			Simulates data from Gaussian deep mixed model with multivariate random effects.
			'data_sim_large.m'		Simulates data from large Gaussian deep mixed model with multivariate random effects.
			'davi_deeplmm_vecRE_train.m'	Estimates Gaussian deep mixed model using DAVI.
			'hybrid_Deeplmm_vecRE_train.m'	Estimates Gaussian deep mixed model using hybrid methods. 
		
        In 'Data' folder:
		'parameters_small.mat'			Contains parameters used to generate data for example 2(a) using 'data_sim.m'.
		'parameters_large.mat'			Contains parameters used to generate data for example 2(b) using 'data_sim_large.m'.


'Example 3'	This folder contains code for the simulation and estimation of a Bernoulli deep mixed model.  
		'Eg3_DeepGLMM.m' 			Script to replicate the results in example 3. 
		'Eg3_DeepGLMM_robust.m'			Script to replicate the boxplot in example 3.
							        These scripts call for functions in the 'Service functions' subfolder.  
		Main functions include:
			'simulate_DeepGLMM.m'	Simulates data from Bernoulli deep mixed model with multivariate random effects.
			'deepGLMMfit.m'			NAGVAC code from Tran et al.(2020), available at 
									https://github.com/VBayesLab/deepGLMM/tree/master/deepGLMM.
			'hybrid_deepglmm_train.m'	Estimates Bernoulli deep mixed model using hybrid methods.  

'Graphs'	This folder contains code to replicate graphs.  
            'lek_profile.m' 	        Script to produce lek profiles for the finance example.

							
Additional notes:
		1. Function 'default_settings.m' generate a structure array that stores default values for hybrid methods.
			SGA = 0: 		using NG-HVI; SGA = 1 indicates using SG-HVI.
			N = 3000: 		number of optimization steps.
			damping = 10: 		damping factor.
			grad_weight = 0.6: 	momentum weight in Step (d) in Algorithm 1.
			J = 20: 		number of MC simulation in predicting step.

