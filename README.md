# Descriptors, experiments and implementation for the submission of the paper: "Understanding the relationship between voice and accompaniment on symbolic music data"


### Abstract
The education of singing and musical instruments in Western music traditions usually draws on piano accompaniment. 
In this paper we study the relationship between this accompaniment and the lead melody on a symbolic corpus 
containing pedagogic-designed piano accompaniments for classical and Rock & Pop singing lessons. We derive 
accompaniment support features on three dimensions: melody, rhythm, and harmony. This allows us to understand the 
role of accompaniment throughout the different levels in which a formal syllabus of music education is structured.
Our analysis shows that the support of accompaniment differs between the classical and Rock & Pop and confirms 
its variability when related to the structure of the syllabus. We develop a web-visualisation tool which allows 
us to explore how the pieces are distributed along the considered dimensions. We complement the discussion with 
a thorough analysis of a set of outlier songs selected using the web-based tool. 

### Dataset
We use a subset of the vocal repertoire of the Voice and Co. Corpus (see [metadata](https://voice-and-co.github.io/corpus)). This subset is based on two collections of scores, with 
Classical and Rock & Pop scores. We have nine different difficulty grades, from Grade 0 to 8 Grade, the latter being 
the hardest one. Since some of these scores present problems in the XML format, we have developed a Sibelius plugin
to properly convert the data to MIDI format for further investigation. We provide plugin in the /plugin folder. 

### Computing the descriptors
Please refer to the .py files in extractor folder to find the source code to compute the proposed descriptors.
The computing functions are called from feature_extraction.py, where the different computing functions are
called.

### Reproducing the experiments
We are attaching a notebook to run experiments. The technical specifications needed to run the experiments are 
well-detailed in the notebook. 
Install the requeriments by:

```
pip install -r requeriments.txt
```

Since preparing the framework may produce issues, and taking into account that the 
syllabi cannot be shared openly (which clearly affects the reproducibility of the experiment), we provide 
already run cells with proper explanation. Hence, the reader is able to understand and examine how the 
experiments are carried out.
