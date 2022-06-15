# Timestamp2Vec


- [About](#about)
- [Explanation](#explanation)
- [Usage](#usage)
- [How to run](#how-to-run)


# About
This Repository is made for the Bachelor Thesis of Gideon Rouwendaal (2022). It contains an explanation about  Timestamp2Vec, information about the usage and information about the VAE and NAS. Timestamp2Vec, the repository, also contains a Visualization and Evaluation Notebook. The Visualization Notebook can be used to visualize the latent space created by the model. The Evaluation Notebook demonstrates how timestamps can be vectorized by Timestamp2Vec. In addition, an evaluation of the model is performed. <br>
In the VAE folder, a VAE Notebook can be found that will create and train a Variational Autoencoder. The encoder of this VAE is used in the Timestamp2Vec model. <br>
In the NAS folder, a NAS Notebook can be found. Neural Architecture Search (NAS) has been used to approximate the  optimal arhicture for the VAE.

# Explanation
The Timestamp2Vec model consists of 2 parts: a feature extractor and an encoder, which is originally part of a variational autoencoder (VAE). The input of Timestamp2Vec is a timestamp in the shape YYYY-MM-DD HH:MM:SS and is of type String. The output of the model is a numpy array of shape (1, 8).
![](./Timestamp2Vec.png)
# Usage
Timestamp2Vec is used to vectorize timestamps. To load the model, import the Timestamp2Vec Class from Timestamp2Vec.py from the Timestamp2Vec_Class folder and create an instance of it. Use the __call__ method to retrieve the embedding.
## Example
```python
from Timestamp2Vec_Class.Timestamp2Vec import *

timestamp2vec = Timestamp2Vec()
vectorized = timestamp2vec("2000-01-01 00:00:00")
print(vectorized)
```

## Output
```
array([[-2.7181404 , -1.6645893 ,  0.06427887, -1.684483  ,  0.0221834 ,
        -0.05639424,  1.3969079 ,  0.705029  ]], dtype=float32)
```

# How to run
1. Clone the Timestamp2Vec repository 
2. Install all the required python packages
```
pip install -r requirements.txt
```
3. (Optional) create the dataset the VAE will be trained on
```
cd ./Timestamp2Vec/Data
python data_creation.py
```
4. (Optional) extract the features from the generated dates (specify the correct file location(s))
```
cd ./Timestamp2Vec/Data
python vectorize_date.py
```
5. (Optional) run the Neural Architectural Search (NAS) Notebook, in order to approximate the optimal parameters of the network (the static architecture provided could be used, but a different architecture could also be explored)
```
cd ./Timestamp2Vec/NAS
jupyter notebook
```
6. (Optional) run the Variational Autoencoder (VAE) Notebook, in order to train the encoder needed for the Timestamp2Vec model. A different architecture for the encoder and decoder could be introduced as opposed to the one in the Notebook. Please specify the required variables and correct file location(s)
```
cd ./Timestamp2Vec/VAE
jupyter notebook
```
7. (Optional) once the VAE has been trained, save the encoder and decoder separately (as coded in the Notebook). Locate the encoder and place the folder of the model in the Timestamp2Vec_Class folder. 
8. Load the model in the file in which the timestamps have to be vectorized
```python
from Timestamp2Vec_Class.Timestamp2Vec import *

timestamp2vec = Timestamp2Vec()
```
9. Use the timestamp2vec object to vectorize the timestamps. Please note that the data types of the to be vectorized timestamps should be Strings.