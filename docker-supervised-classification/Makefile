install:
	pip install --upgrade pip &&\
		pip install -r src/requirements.txt
		
		
format:
	black src/*.py

lint:
	pylint --disable=R,C src/model_training.py

test:  
	python -m pytest -vv --cov=hello test_hello.py 
	
	
all: install lint test