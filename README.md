# Desperately Seeking Experts:	Characterizing Experts for Matching Problems by their Behavior
<p align="center">
<img src ="/feature_extraction.pdf">
</p>

## Prerequisites:
1. [Anaconda 3](https://www.anaconda.com/download/)
2. [Tensorflow (or tensorflow-gpu)](https://www.tensorflow.org/install/)
3. [Keras](https://keras.io/#installation)
4. [Shap](https://github.com/slundberg/shap)

## The Paper
The paper is under review.

## The Team
The environment was developed at the Technion - Israel Institute of Technology by [Roee Shraga](https://sites.google.com/view/roee-shraga/) in collaboration with [Dr. Ofra Amir](https://scholar.harvard.edu/oamir) and [Prof. Avigdor Gal](https://agp.iem.technion.ac.il/avigal/)

| Feature                                    |                                     Description                                    |
|--------------------------------------------|:----------------------------------------------------------------------------------:|
| $\Phi_{LRSM}$\_avg                         |                      the average values in $M$~\cite{Sagi2013}                     |
| $\Phi_{LRSM}$\_max                         |                        maximal value in $M$~\cite{Sagi2013}                        |
| $\Phi_{LRSM}$\_std                         |             the standard deviation of the values in $M$~\cite{Sagi2013}            |
| $\Phi_{LRSM}$\_mpe                         |                  the average entropy over $M$ rows~\cite{lrsmTech}                 |
| $\Phi_{LRSM}$\_mcd                         |     the average difference between $M$ entries and mean entry values~\cite{MCD}    |
| $\Phi_{LRSM}$\_bmc                         |       the distance between $M$ and its closest binary matrix~\cite{Sagi2013}       |
| $\Phi_{LRSM}$\_bmm                         |        similarity between $M$ and its closest binary matrix~\cite{Sagi2013}        |
| $\Phi_{LRSM}$\_lmm                         | similarity between $M$ and its a matrix with a single $1$ in a row~\cite{Sagi2013} |
| $\Phi_{LRSM}$\_dom                         |                ratio of dominant values in $M$~\cite{mao2011towards}               |
| $\Phi_{LRSM}$\_pca1                        |                the first principal component of $M$~\cite{lrsmTech}                |
| $\Phi_{LRSM}$\_pca2                        |                the second principal component of $M$~\cite{lrsmTech}               |
| $\Phi_{LRSM}$\_pcaSum                      |         the sum of informative principal components of $M$~\cite{lrsmTech}         |
| $\Phi_{LRSM}$\_pcaEntropy                  |             the entropy of principal components of $M$~\cite{lrsmTech}             |
| $\Phi_{LRSM}$\_norms1                      |               maximum absolute column sum norm of $M$~\cite{lrsmTech}              |
| $\Phi_{LRSM}$\_norms2                      |                        spectral norm of $M$~\cite{lrsmTech}                        |
| $\Phi_{LRSM}$\_normsF                      |                        frobenius norm of $M$~\cite{lrsmTech}                       |
| $\Phi_{LRSM}$\_normsInf                    |                maximum absolute row sum norm of $M$~\cite{lrsmTech}                |
| $\Phi_{Mou}$\_totalLength                  |                 length of $G$~\cite{rzeszotarski2011instrumenting}                 |
| $\Phi_{Mou}$\_totalActions                 |                number of logged actions in $G$~\cite{goyal2018your}                |
| $\Phi_{Mou}$\_totalTime                    |           $G.T - G.1$~\cite{goyal2018your, rzeszotarski2011instrumenting}          |
| $\Phi_{Mou}$\_totalDist                    |                     total distance in $G$~\cite{goyal2018your}                     |
| $\Phi_{Mou}$\_maxSpeed                     |                      maximum speed in $G$~\cite{wu2016novices}                     |
| $\Phi_{Mou}$\_minX                         |                  the minimal $x$ point in $G$~\cite{wu2016novices}                 |
| $\Phi_{Mou}$\_minY                         |                  the maximal $y$ point in $G$~\cite{wu2016novices}                 |
| $\Phi_{Mou}$\_maxX                         |                  the maximal $x$ point in $G$~\cite{wu2016novices}                 |
| $\Phi_{Mou}$\_maxY                         |                  the maximal $y$ point in $G$~\cite{wu2016novices}                 |
| $\Phi_{Mou}$\_avgSpeed                     |                      average speed in $G$~\cite{wu2016novices}                     |
| $\Phi_{Mou}$\_avgX                         |                  average $x$ location in $G$~\cite{wu2016novices}                  |
| $\Phi_{Mou}$\_avgY                         |                  average $y$ location in $G$~\cite{wu2016novices}                  |
| $\Phi_{Beh}$\_countDistinctCorr            |                  number of distinct element pairs evaluated in $H$                 |
| $\Phi_{Beh}$\_countGeneralCorr             |                   total number of element pairs evaluated in $H$                   |
| $\Phi_{Beh}$\_countMindChange              |                       number of element pairs changed in $H$                       |
| $\Phi_{Beh}$\_avgConf                      |                              average confidence in $H$                             |
| $\Phi_{Beh}$\_maxConf                      |                              maximum confidence in $H$                             |
| $\Phi_{Beh}$\_minConf                      |                              minimum confidence in $H$                             |
| $\Phi_{Beh}$\_avgTime                      |                        average time spent of decision in $H$                       |
| $\Phi_{Beh}$\_maxTime                      |                        maximum time spent of decision in $H$                       |
| $\Phi_{Beh}$\_minTime                      |                        minimum time spent of decision in $H$                       |
| $\Phi_{Seq}$\_conf ($P, R, Res, Cal$)      |          $\|C\|$-size vector of LSTM predictions over confidence sequence          |
| $\Phi_{Seq}$\_time ($P, R, Res, Cal$)      |             $\|C\|$-size vector of LSTM predictions over time sequence             |
| $\Phi_{Seq}$\_consensus ($P, R, Res, Cal$) |           $\|C\|$-size vector of LSTM predictions over consensus sequence          |
| $\Phi_{Spa}$\_Move ($P, R, Res, Cal$)      |             $\|C\|$-size vector of CNN predictions over $P_{\emptyset}$            |
| $\Phi_{Spa}$\_LMouse ($P, R, Res, Cal$)    |                 $\|C\|$-size vector of CNN predictions over $P_{l}$                |
| $\Phi_{Spa}$\_WMouse ($P, R, Res, Cal$)    |                 $\|C\|$-size vector of CNN predictions over $P_{s}$                |
| $\Phi_{Spa}$\_RMouse ($P, R, Res, Cal$)    |                 $\|C\|$-size vector of CNN predictions over $P_{r}$                |