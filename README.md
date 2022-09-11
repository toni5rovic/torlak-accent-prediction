# Torlakian Accent Prediction

## NPFL128 - Language Technologies in Practice Project

Antonije Petrović, Charles University in Prague


# Introduction

Torlakian dialect is a nonstandard dialect of Serbian language, mostly spoken in the South-Eastern region of Serbia, near the border with Bulgaria. This dialect is fairly different from the standard Serbian language to the extent that some speakers from the Western or Northern regions have difficulties understanding speakers from the Torlak area. It differs in many features, some of which are: 
- lexical: more influenced by Turkish and Greek, usage of words that are considered archaic in standard Serbian
- morphology: less complex inflection system - instrumental case merges with genitive case, locative and genitive merge with nominative
- syntax: loss of infinitive, 
- phonology: lack of phoneme /x/, syllabic /l/, appeareance of schwa /ə/, accent position and quality

In the general population, there is a trend to consider this dialect a lesser form of Serbian, with attached stereotypes of it being used by rural, uneducated and old people. Nowadays, young speakers from the area abandon Torlak features and adopt the standard ones instead. This makes Torlak dialect an endangered one.

# Dataset

The corpus used is "Spoken Torlak dialect corpus" available [here](https://www.clarin.si/repository/xmlui/handle/11356/1281) and described in (Vuković, 2021).

The dataset we are working with consists of 168 .txt files with Torlak speakers' utterances only (excluding the researcher's utterances, which may be in the standard form). Transcription included the stress position in the words by using capital letters. 

Example:

```
razgovAra si nAšinski gOvornici
i a On pOsle kƏd si pojdE a On kAže
jAo brE drAgoslave
```

The corpus has been built based on the speech in the Timok area which belongs to the bigger Torlak area.

![Timok area within Torlak area](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs10579-020-09522-4/MediaObjects/10579_2020_9522_Fig1_HTML.png?as=webp)

Even in this smaller area, accent position varies, since speakers occassionally use accent position from the standard Serbian. Examples:
- n**A**pravio <sub>standard Serbian</sub> vs. napr**A**vio <sub>Torlak variation 1</sub> vs. naprav**I**o <sub>Torlak variation 2</sub> eng. `make-PST-1SG-M`
- k**U**tije <sub>standard Serbian</sub> vs. kut**I**je <sub>Torlak variation</sub> eng. `box-PL-NOM`


# Model and Experiments

The implemented model is a simple neural network with dense layers. The input is processed in a sliding window of a set size. The alphabet size configures how much of the most common characters are used, while the rest are replaced by special symbol `<unk>`
 

Baseline accuracy with all lowercased letters in the test data is 86.37%. We will try to achieve a higher score.

| ID   | Epochs | Batch size | Dropout | Alphabet size | Window | Clean | Layers            | Learning rate start/end | Val accuracy | Test accuracy |
| ---- | ------ | ---------- | ------- | ------------- | ------ | ----- | ----------------- | ----------------------- | ------------ | ------------- |
| 0001 | 10     | 256        | 0.3     | 64            | 10     | No    | 256, 128, 64      | 0.001                   | 95.47%       | 97.68%        |
| 0002 | 10     | 256        | 0.5     | 64            | 10     | No    | 256, 128, 64      | 0.001                   | 95.51%       | 97.76%        |
| 0003 | 10     | 256        | 0.3     | 64            | 10     | Yes   | 256, 128, 64      | 0.001                   | 95.40%       | 97.72%        |
| 0004 | 10     | 256        | 0.5     | 64            | 10     | Yes   | 256, 128, 64      | 0.001                   | 95.42%       | 97.83%        |
| 0005 | 15     | 256        | 0.5     | 64            | 12     | Yes   | 256, 128, 64      | 0.001                   | 95.27%       | 97.88%        |
| 0006 | 15     | 256        | 0.5     | 70            | 18     | Yes   | 256, 128, 64      | 0.001                   | 94.92%       | 97.62%        |
| 0007 | 15     | 256        | 0.5     | 75            | 16     | Yes   | 256, 128, 64      | 0.001                   | 95.04%       | 97.74%        |
| 0008 | 15     | 256        | 0.5     | 64            | 12     | Yes   | 512, 256, 128, 64 | 0.001                   | 95.24%       | 94.65%        |
| 0009 | 20     | 64         | 0.5     | 62            | 6      | Yes   | 256, 128, 16      | 0.001 -> 0.0001         | 95.70%       | 97.85%        |
| 0010 | 20     | 128        | 0.5     | 62            | 6      | Yes   | 512, 128, 16      | 0.001 -> 0.0001         | 95.80%       | 98.12%        |
| 0011 | 20     | 64         | 0.5     | 62            | 8      | Yes   | 128, 64, 16       | 0.001 -> 0.0001         | 95.47%       | 97.93%        |
| 0012 | 20     | 512        | 0.4     | 80            | 10     | Yes   | 2048, 512, 16     | 0.001 -> 0.0001         | 95.25%       | 97.55%        |
| 0013 | 20     | 512        | 0.4     | 80            | 10     | Yes   | 2048, 1028, 32    | 0.001 -> 0.0001         | 95.22%       | 97.49%        |

We can see that increasing the width of the network helped and we got test accuracy of 98.12%, 


A few more experiments are run with a higher network width, but no considerable improvement is made.

| ID   | Epochs | Batch size | Dropout     | Alphabet size | Window | Clean | Layers        | Learning rate start/end | Val accuracy | Test accuracy |
| ---- | ------ | ---------- | ----------- | ------------- | ------ | ----- | ------------- | ----------------------- | ------------ | ------------- |
| 0014 | 25     | 128        | 0.5         | 62            | 8      | Yes   | 1024, 256, 32 | 0.001 -> 0.0001         | 95.54%       | 97.81%        |
| 0015 | 25     | 128        | 0.5         | 62            | 6      | Yes   | 1024, 32      | 0.001 -> 0.0001         | 95.73%       | 98.13%        |
| 0016 | 15     | 256        | 0.5         | 62            | 6      | Yes   | 2048, 64      | 0.001 -> 0.0001         | 95.79%       | 98.07%        |
| 0017 | 25     | 256        | 0.3 +0.3 LS | 62            | 6      | Yes   | 2048, 128     | 0.001 -> 0.0001         | 95.54%       | 98.00%        |
| 0018 | 25     | 256        | 0.3 +0.2 LS | 80            | 8      | Yes   | 1024, 128, 32 | 0.001 -> 0.0001         | 95.52%       | 97.61%        |
| 0019 | 20     | 256        | 0.4 +0.2 LS | 70            | 6      | Yes   | 4096, 64      | 0.001 -> 0.0001         | 95.64%       | 98.11%        |
| 0020 | 20     | 256        | 0.5 +0.3 LS | 70            | 8      | Yes   | 2048, 32      | 0.001 -> 0.0001         | 95.55%       | 98.14%        |

Let's take a look at some of the mistakes the model makes.

- More than one accent position in a word:
  - **peškIrčE** pElena
  - kadA se onAj trAktor **IsprAvi**
- Due to the speakers who mix standard and dialect pronounciation, we have mixed data in the corpus. Therefore, the model might not be able to fully capture the "correct" Torlak accent position in every case.

# Future work

It might be interesting to try out different neural architectures and maybe use more data in order to get better performance.

# References

Vuković, T. (2021). Representing variation in a spoken corpus of an endangered dialect: the case of Torlak. Language Resources and Evaluation, 55(3), 731–756. doi:10.1007/s10579-020-09522-4