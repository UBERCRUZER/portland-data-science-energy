# portland-data-science-energy
Portland Data Science Meetup - Energy Dataset

https://www.meetup.com/Portland-Data-Science-Group/events/266223427/

Data was downloaded from https://www.eia.gov/consumption/residential/

This analysis is focused on selecting the best measures to predict whether or not a customer was going experience hardship in paying heating or cooling bills. 

A target variable was created by selecting all "hardship" columns and creating a "hardship score" by using PCA. The first principal component was used as the target for the model.

This approach uses "bestK" method to select the most important features (someone else used a decision tree, which I think would have made the most sense in this context and I didn't want to duplicate efforts). Between bestK and the decision tree, 8/10 of the top features were congruent.
