This curriculum is copied from Melanie Walsh's [*Introduction to Cultural Analytics & Python*](https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html). A few code updates were made to reflect the newest versions of Scikit-learn and Pandas. 

The dataset for the curriculum is available [here](https://melaniewalsh.github.io/Intro-Cultural-Analytics/_downloads/64e2547e2d86c20cc2a74f660143cfeb/US_Inaugural_Addresses.zip). 

# TF-IDF with Scikit-Learn

In the previous lesson, we learned about a text analysis method called *term frequency–inverse document frequency*, often abbreviated *tf-idf*. Tf-idf is a method that tries to identify the most distinctively frequent or significant words in a document. We specifically learned how to calculate tf-idf scores using word frequencies per page—or "extracted features"—made available by the HathiTrust Digital Library.

In this lesson, we're going to learn how to calculate tf-idf scores using a collection of plain text (.txt) files and the Python library scikit-learn, which has a quick and nifty module called [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

In this lesson, we will cover how to:
- Calculate and normalize tf-idf scores for U.S. Inaugural Addresses with scikit-learn

## Dataset

### U.S. Inaugural Addresses

<blockquote class="epigraph" style=" padding: 10px">

This is the meaning of our liberty and our creed; why men and women and children of every race and every faith can join in celebration across this magnificent Mall, and why a man whose father less than 60 years ago might not have been served at a local restaurant can now stand before you to take a most sacred oath.  So let us mark this day with remembrance of who we are and how far we have traveled.
<p class ="attribution">—Barack Obama, Inaugural Presidential Address, January 2009
    </p>
    
</blockquote>

During Barack Obama's Inaugural Address in January 2009, he mentioned "women" four different times, including in the passage quoted above. How distinctive is Obama's inclusion of women in this address compared to all other U.S. Presidents? This is one of the questions that we're going to try to answer with tf-idf.

## Breaking Down the TF-IDF Formula

But first, let's quickly discuss the tf-idf formula. The idea is pretty simple.

**tf-idf = term_frequency * inverse_document_frequency**

**term_frequency** = number of times a given term appears in document

**inverse_document_frequency** = log(total number of documents / number of documents with term) + 1**\***

You take the number of times a term occurs in a document (term frequency). Then you take the number of documents in which the same term occurs at least once divided by the total number of documents (document frequency), and you flip that fraction on its head (inverse document frequency). Then you multiply the two numbers together (term_frequency * inverse_document_frequency).

The reason we take the *inverse*, or flipped fraction, of document frequency is to boost the rarer words that occur in relatively few documents. Think about the inverse document frequency for the word "said" vs the word "pigeon." The term "said" appears in 13 (document frequency) of 14 (total documents) *Lost in the City* stories (14 / 13 --> a smaller inverse document frequency) while the term "pigeons" only occurs in 2 (document frequency) of the 14 stories (total documents) (14 / 2 --> a bigger inverse document frequency, a bigger tf-idf boost). 

*There are a bunch of slightly different ways that you can calculate inverse document frequency. The version of idf that we're going to use is the [scikit-learn default](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer), which uses "smoothing" aka it adds a "1" to the numerator and denominator: 

**inverse_document_frequency**  = log((1 + total_number_of_documents) / (number_of_documents_with_term +1)) + 1

<div class="margin sidebar" style=" padding: 10px">

> If smooth_idf=True (the default), the constant “1” is added to the numerator and denominator of the idf as if an extra document was seen containing every term in the collection exactly once, which prevents zero divisions: idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1.  
> -[scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)

</div>

## TF-IDF with scikit-learn

[scikit-learn](https://scikit-learn.org/stable/index.html), imported as `sklearn`, is a popular Python library for machine learning approaches such as clustering, classification, and regression. Though we're not doing any machine learning in this lesson, we're nevertheless going to use scikit-learn's `TfidfVectorizer` and `CountVectorizer`.

Install scikit-learn


```python
!pip install sklearn
```

Import necessary modules and libraries


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
pd.options.display.max_rows = 600
from pathlib import Path  
import glob
```

We're also going to import `pandas` and change its default display setting. And we're going to import two libraries that will help us work with files and the file system: [`pathlib`](https://docs.python.org/3/library/pathlib.html##basic-use) and [`glob`](https://docs.python.org/3/library/glob.html).

#### Set Directory Path

Below we're setting the directory filepath that contains all the text files that we want to analyze.


```python
directory_path = "US_Inaugural_Addresses/"
```

Then we're going to use `glob` and `Path` to make a list of all the filepaths in that directory and a list of all the short story titles.


```python
text_files = glob.glob(f"{directory_path}/*.txt")
```


```python
text_files
```


```python
text_titles = [Path(text).stem for text in text_files]
```


```python
text_titles
```

## Calculate tf–idf

To calculate tf–idf scores for every word, we're going to use scikit-learn's [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

When you initialize TfidfVectorizer, you can choose to set it with different parameters. These parameters will change the way you calculate tf–idf.

The recommended way to run `TfidfVectorizer` is with smoothing (`smooth_idf = True`) and normalization (`norm='l2'`) turned on. These parameters will better account for differences in text length, and overall produce more meaningful tf–idf scores. Smoothing and L2 normalization are actually the default settings for `TfidfVectorizer`, so to turn them on, you don't need to include any extra code at all.

Initialize TfidfVectorizer with the following parameters: input = 'filename' because we are iterating across a list of files, and stop_words='english' to apply the english language list of stop words


```python
tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')
```

Run TfidfVectorizer on our `text_files` and apply the fit_transform function to convert the data into a matrix format necessary to perform the tf-idf calculations


```python
tfidf_vector = tfidf_vectorizer.fit_transform(text_files)
```


```python
type(tfidf_vector)
```

Make a DataFrame out of the resulting tf–idf vector, applying the `toarray()` function to transform the matrix into a format that can become a dataframe, setting the index equal to the list of titles, and the columns equal "feature names" or words as columns using the `tfidf_vectorizer.get_feature_names_out()` function


```python
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names_out())
```

Add column for document frequency to note the number of times word appears in all documents


```python
tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
```

create a slice of the dataframe to target specific words that are of interest. Then sort the sliced dataframe by the index and round the decimals to two

```python
tfidf_slice = tfidf_df[['government', 'borders', 'people', 'obama', 'war', 'honor','foreign', 'men', 'women', 'children']]
tfidf_slice.sort_index().round(decimals=2)
```

Let's drop "OO_Document Frequency" since we were just using it for illustration purposes.


```python
tfidf_df = tfidf_df.drop('00_Document Frequency', errors='ignore')
```

Use the stack function to reorganize the DataFrame so that the words are in rows rather than columns. We need to reset the index to make sure the results became a dataframe and not a series object


```python
tfidf_df.stack().reset_index()
```


```python
tfidf_df = tfidf_df.stack().reset_index()
```

Rename our columns

```python
tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term'})
```

To find out the top 10 words with the highest tf–idf for every story, we're going to sort by document and tfidf score and then groupby document and take the first 10 values.


```python
tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
```


```python
top_tfidf = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(10)
```

We can zoom in on particular words and particular documents.


```python
top_tfidf[top_tfidf['term'].str.contains('women')]
```

It turns out that the term "women" is very distinctive in Obama's Inaugural Address.


```python
top_tfidf[top_tfidf['document'].str.contains('obama')]
```


```python
top_tfidf[top_tfidf['document'].str.contains('trump')]
```


```python
top_tfidf[top_tfidf['document'].str.contains('kennedy')]
```

## Visualize TF-IDF

We can also visualize our TF-IDF results with the data visualization library Altair.


```python
# !pip install altair
```

Let's make a heatmap that shows the highest TF-IDF scoring words for each president, and let's put a red dot next to two terms of interest: "war" and "peace":

The code below was contributed by [Eric Monson](https://github.com/emonson). Thanks, Eric!


```python
import altair as alt
import numpy as np

# Terms in this list will get a red dot in the visualization
term_list = ['war', 'peace']

# adding a little randomness to break ties in term ranking
top_tfidf_plusRand = top_tfidf.copy()
top_tfidf_plusRand['tfidf'] = top_tfidf_plusRand['tfidf'] + np.random.rand(top_tfidf.shape[0])*0.0001

# base for all visualizations, with rank calculation
base = alt.Chart(top_tfidf_plusRand).encode(
    x = 'rank:O',
    y = 'document:N'
).transform_window(
    rank = "rank()",
    sort = [alt.SortField("tfidf", order="descending")],
    groupby = ["document"],
)

# heatmap specification
heatmap = base.mark_rect().encode(
    color = 'tfidf:Q'
)

# red circle over terms in above list
circle = base.mark_circle(size=100).encode(
    color = alt.condition(
        alt.FieldOneOfPredicate(field='term', oneOf=term_list),
        alt.value('red'),
        alt.value('#FFFFFF00')        
    )
)

# text labels, white for darker heatmap colors
text = base.mark_text(baseline='middle').encode(
    text = 'term:N',
    color = alt.condition(alt.datum.tfidf >= 0.23, alt.value('white'), alt.value('black'))
)

# display the three superimposed visualizations
(heatmap + circle + text).properties(width = 600)
```
