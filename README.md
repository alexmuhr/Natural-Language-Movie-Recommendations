# Natural-Language-Movie-Recommendations  
Movie recommendations based on user written passages about preferred movies. Final project for Metis Immersive Data Science Program. Example app built on this project is featured [here](http://18.218.199.101/).
  
---  
## Overview
Modern media recommendations within services such as Netflix and Spotify follow a shotgun approach. User's are presented large quantities of recommendations based on media that they have previously consumed and rated. Although a user is likely to enjoy the presented recommendations, these recommendations may not reflect what a user wants at the specific moment. The shotgun approach may also cause decision paralysis wherein a user is overwhelmed by the quantity of choices.  
  
The goal of this project is to build a recommendation engine based on a user's written passage about their preferences. In this way a user can describe specifically what they are looking for and recieve a recommendation that follows suit.

This Github repo contains 3 files that are used to power a movie recommendation engine based on natural language user inputs:
1. movies_dataset_engineering.ipynb: This notebook file is used to load in [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/downloads/the-movies-dataset.zip/7#movies_metadata.csv) from Kaggle and create a variety of pickle objects used to power the recommendation engine.
2. entity_sentiment.py: This file contains a variety of functions used to determine sentiment towards a specific entity mentioned in text. In this case the entities are intended to be movies, actors, and genres, however these functions are agnostic towards the movie domain and could be applied more broadly. The functions leverage the [Stanford Dependency Parser](https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip) and Vader Sentiment package.
3. recommendations.py: This file contains functions used to generate recommendations. It imports entity_sentiment.py and uses the contained functions to establish user preferences. Movies are compared based on similarity of their synopses and thematic keywords. Facet searching is incorporated for strongly liked or disliked actors and genres.
  
## Using this Repo
In order to recreate this project follow the steps outlined below:
1. Download [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/downloads/the-movies-dataset.zip/7#movies_metadata.csv) from Kaggle, unzip, and place the files within your project in a folder named the-movies-dataset.
2. Install the required libraries: nltk, vaderSentiment, gensim, sklearn, numpy, pandas, etc. The Stanford Dependency Parser must be downloaded from this [link](https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip) and requires a java installation. You must provide the path to your java installation and the stanford dependency parser files within entity_sentiments.py.
3. Run movies_dataset_engineering.ipynb to load The Movies Dataset and create pickle objects that allow the recommendation engine to operate quickly and with minimal RAM. Afterwards the original The Movies Dataset files can be deleted.
4. Open a python notebook or terminal within your project directory and import recommendations.py. Calling the function `recommendations.full_rec_pipeline('some text about movies')` will generate a string that acknowledges your preferences and suggests a movie to watch.
  
#### OR
  
Use the functions from entity_sentiment.py for your own project. These functions are agnostic to movies and recommendations. They possess much broader applications and could be used for any project that requires determining sentiment towards specific people, places, or things.
