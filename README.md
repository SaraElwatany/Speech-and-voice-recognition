# Speech And Voice Recognition System "GMM Model"

# Abstract
The above project was a collaborative university project, where we used "GMM" to recognize the speech patterns for different words and the voice of the team's members. Where for the "Voice Recognition" part, we recorded about 5 samples for each group member with the write sentence to say, "Open the door", and then we built our models for each member after extracting the features from them "FeatureExtraction.py", to get the write person we used the loglikelihood as our similarity score, where for each prediction we get the similarity for each model and then we assign the user of the application to the member corresponding to the highest similarity score from the models but only after comparing with a certain thresholds (bounds), otherwise the person won't be allowed to have "Access".  Where for the "Speech Recognition" part, we recorded about 15 samples for each wrong sentence, "Open the book", "Open the window", "Please open", and then we seeked the same approach as above for each wrong sentence by making a model followed by similarity score calculation for each trial and then comparing the similarity score with certain thresholds.

## Snapshots
- Correct Sentence & User
![2023-06-19 (6)](https://github.com/SaraElwatany/Speech-and-voice-recognition/assets/93448764/18a6c093-32e9-4a3a-9919-0abbde7512ef)

- Correct Sentence but wrong word
![2023-06-19 (7)](https://github.com/SaraElwatany/Speech-and-voice-recognition/assets/93448764/bfc09576-4dab-4060-8201-195578b24147)

## Note
- Don't forget to modify the paths present in the main.py to match your own.
- The "svm1.py" and "svm2.py" files were only for our own tests and has nothing to do with running the project.

## Languages Used
- Python
- HTML
- CSS

## How to run
- Clone the repository
- Run main.py
  

  
