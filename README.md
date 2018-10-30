# Random Boosting

Random Boosting is an algorithm I have developed for my master's thesis. The central idea is that RB builds on Friedman's Gradient Boosted Trees, but adds a new random component to the boosting procedure concerning the depth of a tree. More specifically, at each iteration, a random number between 1 and some upper limit is drawn that determines the maximally possible depth a tree can have at a certain step.

The algorithm is developed based on `sklearn.ensemle.GradientBoostingRegressor` and `sklearn.ensemle.GradientBoostingClassifier`and is used in exactly the same way (i.e. argument names match excactly and CV can be carried out with `sklearn.model_selection.GridSearchCV`). The only difference is that the `RandomBoosting*`-object uses `max_depth` to randomly draw tree depths for each iteration. All drawn tree depths are stored in the attribute `depths`. You can therefore take a glimpse at the tree sizes via

```python
rb = RandomBoostingRegressor()
rb.fit(...)

# Tree depths
rb.depths
```

`simulation.py` contains an example of how to use Random Boosting in regression. 

Note that you can also use Random Boost by typing `GradientBoostingRegressor(random_depth=True)` or `GradientBoostingClassifier(random_depth=True)` (In fact, I implemented Random Boost as a sub- and wrapper class of the respective Gradient Boosting classes), which makes it usable as a simple add-on just like Stochastic Gradient Boosting.

Please feel free to test and give feedback on the algorithm. Also don't hesitate to contact me if you feel like it. 

FYI: 
I was issuing a [feature request](https://github.com/scikit-learn/scikit-learn/issues/12472) to the scikit-learn package so that Random Boost becomes part of the package. Understandably, the maintaining team won't include it until it becomes mature and popular enough. I deem maturity not to be an issue as it is just a small alteration of the code base, and I will now work on the popularity part by writing a paper. So stay tuned!
