To setup the documentation in the first place we followed: https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/
which includes running 
```bash
cd docs/
sphinx-apidoc -o source/ --ext-autodoc ../src/CompartmentalSystems
```
to create the ```*.rst``` files of the modules in doc/source.
This step has to be repeated after new models (or other modules) have been added.
At the moment the CARDAMON model is excluded since sphinx cannot import it.


To update the documentation run 
```bash
make html
``` 
or 
```bash
sphinx-build -b html source/ build/
```
here.
