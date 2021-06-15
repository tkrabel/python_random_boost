# Random Boosting

[Random Boosting](https://arxiv.org/abs/2009.06078) builds on Friedman's Gradient Boosted Trees, but adds a new random component to the boosting procedure concerning the depth of a tree. More specifically, at each iteration, a random number between 1 and some upper limit is drawn that determines the maximally possible depth a tree can have at a certain step.

The algorithm is developed based on `sklearn.ensemle.GradientBoostingRegressor` and `sklearn.ensemle.GradientBoostingClassifier`and is used in exactly the same way (i.e. argument names match excactly and CV can be carried out with `sklearn.model_selection.GridSearchCV`). The only difference is that the `RandomBoosting*`-object uses `max_depth` to randomly draw tree depths for each iteration.

```python
rb = RandomBoostingRegressor()
rb.fit(...)
rb.predict()
```

`simulation.py` contains an example of how to use Random Boosting in regression. 

Note that you can also use Random Boost by typing `GradientBoostingRegressor(random_depth=True)` or `GradientBoostingClassifier(random_depth=True)` (In fact, I implemented Random Boost as a sub- and wrapper class of the respective Gradient Boosting classes), which makes it usable as a simple add-on just like Stochastic Gradient Boosting.

Please feel free to test and give feedback on the algorithm. Also don't hesitate to contact me if you feel like it. 

## Links

- [Random Boost on arXiv.org](https://arxiv.org/abs/2009.06078)
