# Random Boosting

Random Boosting is an algorithm I have developed for my master's thesis. The central idea is that RB builds on Friedman's MART, but adds a new random component to the boosting procedure concerning the depth of a tree. More specifically, at each iteration, a random number between 1 and some upper limit is drawn that determines the maximally possible depth a tree can have at a certain step.

The algorithm is developed based on `sklearn.ensemle.GradientBoostingRegressor` and is used in exactly the same way (i.e. argument names match excactly and CV can be carried out with `sklearn.model_selection.GridSearchCV`). The only difference is that the `RandomBoostingRegressor`-object uses `max_depth` to randomly draw tree depths for each iteration. All drawn tree depths are stored in the attribute `depths`.

`example.py` contains an example of how to use Random Boosting.

Please feel free to test and give feedback on the algorithm. Also don't hesitate to contact me if you feel like it. 