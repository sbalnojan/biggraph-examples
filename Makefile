del_example_1: ## delete relevant files created by example 1
	cd data/example_1; rm -r dictionary.json entit* t*

del_example_2: ## delete relevant files created by example 2
	cd data/example_2; rm -r dictionary.json entit* t*

del_models: ## delete checkpoints
	rm -r model/example_1; rm -r model/example_2
