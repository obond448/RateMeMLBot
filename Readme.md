# RateMe Machine Learning Bot by Otto Bond

This bot uses a pytorch squeezenet neural network that has undergone finetune training on the CelebA (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to classify face portraits into "attractive" and "not attractive" categories. 

The bot retrieves images from the /r/RateMe subreddit using PRAW and identifies faces in the images using opencv2 face detetion, specified by "haarcascade_frontalface_default.xml". Then rates all detected face images, using the custom NN, and uploads a summary image to the /r/RateMeMLRater subreddit.

The bot is hosted on heroku and run automatically at specified times.

## File Guide


## Interpretation of number output

Numbers at each detected face can be interpreted as the model's determined likelihood that that face is in the "attractive" set.

## Limitations

Predictions by the NN are unreliable for many reasons:
* Faces must be front-on, as they are in the CelebA dataset, or there is a significant efficacy dropoff
* The underlying dataset is biased:
    * It was created by humans
    * It only looks at celebrities
    * The final criteria described by the data, is therefore: Would the person classifying the data believe this picture to show a "attractive" or a "not attractive" celebrity?
* The final model acheives only a ~80% accuracy classifying images on a validation set

