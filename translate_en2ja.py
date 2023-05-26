from transformers import pipeline

def translate(texts):

    fugu_translator = pipeline('translation', model='staka/fugumt-en-ja')
    
    translated = fugu_translator(texts)
    res = []
    for x in translated:
        res.append(x[0]['translation_text'])
    
    return res
    