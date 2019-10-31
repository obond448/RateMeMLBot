0806 - Made using feature extraction on the pretrained squeeze net,
		used 30000 training samples for training set
		10000 for validation, best model achieved ~76% accuracy
	Notes: this is the second model created.
		   took 210 mins to train, 3 epochs
		   
1206 - Used feature extraction only on females
		achieved a accuracy of ~79%
		otherwise same as above
		
1306 - Used fine tuning (all weigths) on females only (almost full dataset)
		achieve a val accuracy of 81.1%
