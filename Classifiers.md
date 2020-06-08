| Classifier |                                   (Short) Description                                  |
|---------------------|:--------------------------------------------------------------------------------------:|
| K Nearest Neighbors [1] |    prediction is based on the k nearest train instances to classify a test instance    |
| Linear SVM [2]         |      prediction is based on a hyperplane aiming to separate the data into classes      |
| RBF SVM [3]            |             prediction is based on SVM with a radial basis function kernel             |
| Gaussian Process [4]   |         prediction is based on a Gaussian approximation over the training data         |
| Decision Tree [5]      |        prediction is based on a tree with nodes corresponding to feature splits        |
| Random Forest [6]      |                  prediction is based on an ensemble of decision trees                  |
| MLP [7]                |      prediction is based on a multi layer perceptron trained using backpropagation     |
| Naive Bayes [8]        |      prediction is based on the Bayes theorem (prior and posterior probabilities)      |
| AdaBoost [9]           | prediction is based on adaptive boosting, treating the input features as weak learners |

Implementations were taken from [scikit-learn](https://scikit-learn.org/stable/)  

**References:**  
[1] Altman, Naomi S. "An introduction to kernel and nearest-neighbor nonparametric regression." The American Statistician 46.3 (1992): 175-185.
[2] Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20.3 (1995): 273-297.
[3] Chang, Yin-Wen, et al. "Training and testing low-degree polynomial data mappings via linear SVM." Journal of Machine Learning Research 11.Apr (2010): 1471-1490.
[4] Rasmussen, Carl Edward. "Gaussian processes in machine learning." Summer School on Machine Learning. Springer, Berlin, Heidelberg, 2003.
[5] Breiman, Leo, et al. Classification and regression trees. CRC press, 1984.
[6] Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
[7] Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation. No. ICS-8506. California Univ San Diego La Jolla Inst for Cognitive Science, 1985.
[8] Friedman, Nir, Dan Geiger, and Moises Goldszmidt. "Bayesian network classifiers." Machine learning 29.2-3 (1997): 131-163.
[9] Freund, Yoav, and Robert E. Schapire. "Experiments with a new boosting algorithm." icml. Vol. 96. 1996.